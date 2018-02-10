from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib.keras.api.keras import backend as Kb

import settings
import statistics as st

from models.specs import my_losses
from models.specs.random_boxes import tf_format_to_abs

from lib import helpers as utils
from lib.show_images import debugShowBoxes

from net_structure import NetworkStructure


def debug_pipe(train_pipe):
    for i in range(1200):
        batch = train_pipe.pull_data()
        words = batch['meta_image'][0].word_list
        debugShowBoxes(batch['image'][0, :].astype(np.uint8) / 255. / 255., boxes=batch['gt_boxes'][:, 1:], titles=words)


def run(train_iterator, train_iters, P, experiment_dir='./', log_steps=30, save_steps=1000, train_mode=True, segmentation_free=False,
        seed=128, num_producer_threads=1):

    np.random.seed(seed)
    tf.set_random_seed(seed)
    build_phocs = P.phoc_dim > 0

    experiment_dir, logger = settings.get_exp_dir_and_logger(experiment_dir)
    train_pipe = settings.get_pipeline(P, train_iterator, num_producer_threads, augmentations=train_mode, crop_words=P.crop_words)

    network = NetworkStructure(P, experiment_dir)

    summary_op = tf.summary.merge_all()
    sess = settings.get_session(P)

    with sess.as_default():
        tb_writer = utils.TensorBoardFiles(experiment_dir, P.log_prefix, sess)

        # Init all variables
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        logger('Initialized vars')

        network.models.load(sess)

        # Global Step reset
        if P.reset_gs:
            gss = []
            if hasattr(network, 'gs_reg'):
                gss.append(network.gs_reg)
            if hasattr(network, 'gs_hmap'):
                gss.append(network.gs_hmap)
            sess.run(tf.variables_initializer(gss))

        save_path = experiment_dir / ('eval' if P.stat_prefix is None else P.stat_prefix if P.eval_run else 'train')

        tf_op_timer = utils.Timer()
        pipe_timer = utils.Timer()
        stats_timer = utils.Timer()
        av_losses = utils.RunningAverages(num_of_averages=len(my_losses()), max_length=log_steps)
        runners = []
        NORMALIZE = P.image_normalize_const

        if train_mode:
            if P.train_hmap:
                runners += [(network.train_hmap, 'hmap', network.gs_hmap, sess.run(network.gs_hmap))]
            if P.train_regression and train_mode:
                runners += [(network.train_boxes, 'regression', network.gs_reg, sess.run(network.gs_reg))]
        else:
            runners += [(network.train_hmap, 'eval', network.gs_hmap, sess.run(network.gs_hmap))]

        print ('Goiong for %d runners' % len(runners))
        for train_op, train_mode, global_step, strt_iter in runners:

            logger('Starting %s from: %d to: %d' % (train_mode, strt_iter, train_iters + 1))
            strt_iter = 0 if P.eval_run else strt_iter
            train_type = train_mode if train_mode else 'Eval'

            # Setting-up tf output ops
            execution = {'gs': global_step, 'losses': my_losses(), 'good_boxes': network.good_boxes}
            if train_mode:
                execution['train_op'] = train_op
                execution['random_boxes'] = network.rnd_boxes if segmentation_free else network.pool_boxes
                if segmentation_free:
                    execution['random_iou_labels'] = network.rnd_iou_labels
            else:
                execution['update_os'] = network.update_ops
            if build_phocs:
                execution['good_phocs'] = network.good_phocs

            for i in range(strt_iter, train_iters + 1):
                # Pull data
                pipe_timer.tic()
                batch = train_pipe.pull_data()
                if batch is None:
                    break

                # Normalize image
                original_image = batch['image'].copy()
                batch['image'] = batch['image'].astype(np.float32) / NORMALIZE

                feed_dict = settings.feed_dict_from_dict(network.inputs, batch, train_pipe, P, train_mode=True)
                feed_dict.update({Kb.learning_phase(): 1*(train_mode)})

                pipe_timer.toc()

                # Train
                tf_op_timer.tic()
                res = sess.run(execution, feed_dict)
                tf_op_timer.toc()

                gs = res['gs']
                # Update Running averages
                av_losses.update(res['losses'])

                # Log steps
                stats_timer.tic()
                if i % log_steps == 0 or not train_mode:

                    logger('-%6d / %6d- GS [%6d] DataTime [%4.2fs] GPUTime [%4.2fs] StatsTime [%4.2fs]-%s [%s]-' %
                           (i, train_iters, gs, pipe_timer.average(), tf_op_timer.average(), stats_timer.average(), train_type, P.name))
                    # Print out loss names and average values
                    logger(' '.join(['%s [%5.4f]' % (v, w()) for v, w in zip([x.name.split('/')[0] for x in my_losses()], av_losses())]))

                    # Evaluation Run statistics
                    if not train_mode:
                        # get boxes with their scores
                        good_boxes_pred = res['good_boxes']
                        abs_good_boxes_pred = tf_format_to_abs(good_boxes_pred, P.target_size)

                        # filter boxes
                        logger('-%6d- BOXES [%4d] DataTime [%4.2fs] GPUTime [%4.2fs] StatsTime [%4.2fs] -EVAL-' %
                               (i, good_boxes_pred.shape[0], pipe_timer.average(), tf_op_timer.average(), stats_timer.average()))

                        if build_phocs:
                            # NOTICE: For PHOCs, only single batch eval is supported
                            box_viz_img = st.update_phoc_stats(meta_images=batch['meta_image'], doc_images=original_image, pred_boxes=abs_good_boxes_pred,
                                                               pred_phocs=res['good_phocs'], gt_boxes=batch['gt_boxes'], save_path=save_path)
                        else:
                            box_viz_img = st.update_segmentation_stats(meta_images=batch['meta_image'], doc_images=original_image, gt_boxes=batch['gt_boxes'],
                                                                       pred_boxes=abs_good_boxes_pred, params=P, save_path=save_path,
                                                                       test_phase=not train_mode, viz=True)
                        if box_viz_img is not None:
                            feed_dict.update({network.inputs.box_viz_images: box_viz_img})

                    else:
                        rboxes = res.get('random_boxes')
                        rlabels = res.get('random_iou_labels', np.array([P.box_filter_num_clsses - 1]*rboxes.shape[0]))
                        rboxes = tf_format_to_abs(rboxes, P.target_size)
                        box_viz_img_tensor = st.train_viz(batch, rboxes, rlabels, phoc_lab_thresh=3, unnormalize=NORMALIZE)

                        if box_viz_img_tensor is not None:
                            feed_dict.update({network.inputs.box_viz_images: box_viz_img_tensor})

                    # Do another pass to log newly create visualizations to TensorBoard
                    summary_protobuf, gs = sess.run([summary_op, global_step], feed_dict)
                    tb_writer.real.add_summary(summary_protobuf, global_step=gs)

                # Save steps
                if i % save_steps == 0 and train_mode:
                    network.models.save(sess, global_step)
                stats_timer.toc()

        # Won't be prefixed, saved as 'model'
        if train_mode and len(runners) > 0:
            network.models.save(sess, global_step)

        # statistics.final_stats()
        logger.close()


def experiment_setup(base_dir, args, passed_params=None):
    """ Setting-up experiment"""

    if passed_params is not None and not isinstance(passed_params, dict):
        raise AttributeError('passed_params must be a dictionary')

    os.environ['CUDA_VISIBLE_DEVICES'] = "%d" % args.gpu_id


    # TODO: move this to be with the rest of arguments defenitions
    # NOTICE: by default we support 5 or 2 classes of IoU boxes.
    # In-case you wish to use other class number you should carefully consider:
    #   (1) Number of boxes per class you wish to generate during training
    #   (2) Lower IoU bound for box proposal classes definition

    box_filter_num_clsses = args.box_filter_num_clsses
    iou_cls_lower_bound =  (0.35 if box_filter_num_clsses == 5 else 0.2) if args.iou_cls_lower_bound is None else args.iou_cls_lower_bound
    boxes_per_class = [50, 50, 100, 100, 100] if box_filter_num_clsses == 5 else [250, 150]

    passed_params = {'boxes_per_class': boxes_per_class, 'iou_cls_lower_bound':iou_cls_lower_bound} if passed_params is None else passed_params

    logger = utils.Logger(log_dir=base_dir)
    passed_params = settings.write_params_to_args(params=passed_params, args=args, override=False)
    settings.log_params(logger, passed_params, None)
    logger.close()

    it = settings.get_dataset_loader(passed_params)

    run(train_iterator=it,
        train_iters=args.iters,
        P=passed_params,
        experiment_dir=base_dir,
        train_mode=not args.eval_run,
        segmentation_free=args.segment_free,
        log_steps=args.log_steps,
        save_steps=args.save_steps,
        )


if __name__ == '__main__':
    import os, time
    from pathlib2 import Path
    from settings.options import BoxSegmentWithPHOCOptions

    args = BoxSegmentWithPHOCOptions().parse()

    base_dir = Path(args.experiment_dir) / args.name
    if not base_dir.exists():
        base_dir.mkdir(parents=True)
    print ('model will be loaded from %s' % str(base_dir))
    print(os.getcwd())
    time.sleep(1)
    experiment_setup(str(base_dir), args, passed_params=None)
