from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import settings

from models.specs import get_update_ops, keep_my_loss, my_losses, ModelsSaveLoadManager
from models.specs.heatmap import FeatureMap, HeatMap
from models.specs.heatmap import heatmap_loss_xent, heatmap_loss_sigmoid
from models.specs.unet import reconstruction_loss, UnetSmoother
from models.specs.regression import reg_loss, BoxRegression
from models.specs import word_embeddings

from models.specs.roi_pooling_features import iou_loss, phoc_loss_func, IoUPrediction
from models.specs.random_boxes import random_boxes_ops
from models.specs import TFNetwork

from settings import get_train_op

reduction = tf.reduce_mean

def get_word_embedding_model(name):
    """ Helper to select a word embedding model from command line argument"""
    assert hasattr(word_embeddings, name), '%s does not exists' % name
    ModelCls = getattr(word_embeddings, name)
    assert issubclass(ModelCls, TFNetwork), '%s is not a Valid Word Embedding model' % name
    return ModelCls


class NetworkStructure(object):
    """
    We have two modes of operation:
        * one step heatmap or two step (additional U-net smoother for the heatmap)
        * segmentation-free - predict word bounding box locations using a regression network (otherwise we assume gt_boxes are available
                              this is used to evaluate PHOC in segmentation based scenario)
        * with PHOCs or w.o. PHOCs
        *
    """

    def __init__(self, P, experiment_dir):
        self.P = P
        self.exp_dir = experiment_dir
        segmentation_free = P.segment_free
        build_phocs = P.phoc_dim > 0

        self.inputs = inputs = settings.get_inputs(P.batch_size, P.target_size, phoc_dim=P.phoc_dim)
        # Helper to manage diffrenet parts of the model
        self.models = models = ModelsSaveLoadManager(exp_dir=experiment_dir)
        input_img = inputs.images

        feature_map = FeatureMap(exp_dir=experiment_dir, args=P, scope='fmap')
        models.add_model(feature_map)

        # (1) Features
        fmap_input = input_img
        features = feature_map.small(fmap_input)

        # (2) Heatmap
        out_cls = 3 if P.hmap_border else 1
        heatmap = HeatMap(output_size=out_cls, scope='hmap', args=P, exp_dir=experiment_dir)
        models.add_model(heatmap)
        hmap_logits = heatmap.heatmap(features)
        hmap = tf.nn.sigmoid(hmap_logits, name='probability_map') if not P.hmap_border else tf.nn.softmax(hmap_logits, dim=-1, name='probability_map')
        if P.hmap_border:
            L_hmap = heatmap_loss_xent(y=inputs.gt_heatmap, y_hat=hmap_logits, weight_pos=P.heatmap_pos_cls_weight,
                                    weight_neg=P.heatmap_neg_cls_weight, with_border=P.hmap_border)
        else:
            hmap_logits_for_loss = hmap_logits
            L_hmap = heatmap_loss_sigmoid(y=inputs.gt_heatmap, y_hat=hmap_logits_for_loss, weight_pos=P.heatmap_pos_cls_weight,
                                  weight_neg=P.heatmap_neg_cls_weight, reduction=reduction)
        keep_my_loss(L_hmap)

        # (2a) Heatmap Smoothing U-NET
        if P.unet_size > 0:
            hmap_smoother = UnetSmoother(out_size=out_cls, size=P.unet_size, down_layers=P.unet_depth, scope='hmap_smoother', args=P, exp_dir=experiment_dir)
            models.add_model(hmap_smoother)
            smoother_input = hmap if not P.unet_with_img else tf.concat([hmap, input_img], axis=-1, name='unet_input')
            smooth_logits = hmap_smoother.unet(smoother_input)
            if P.hmap_border:
                self.smooth_hmap = smooth_hmap = tf.nn.softmax(smooth_logits, dim=-1, name='smoother_prob_map')
                L_smooth = heatmap_loss_xent(y=inputs.gt_heatmap, y_hat=smooth_logits, weight_pos=1., weight_neg=1., with_border=P.hmap_border)
            else:
                # NOTE: In case of binary heatmap we use L1 reconstruction loss
                self.smooth_hmap = smooth_hmap = tf.nn.sigmoid(smooth_logits, name='smoother_prob_map')
                L_smooth = reconstruction_loss(y=inputs.gt_heatmap, y_hat=smooth_hmap, name='L_smoother')

            keep_my_loss(L_smooth)

        # (4) Regression
        if segmentation_free:
            regression = BoxRegression(scope='box_reg', exp_dir=experiment_dir, args=P)
            models.add_model(regression)

            # Do we use smoother? (we do if unet size is positive)
            reg_hmap_input = smooth_hmap if P.unet_size > 0 else hmap
            # Do we even use the heatmap? (We only skip the heatmap for ablation analysis)
            reg_input = input_img if P.hmap_ablation else reg_hmap_input

            reg_features = regression.features(reg_input)
            pred_shifts = regression.box_shifts(reg_features)
            batched_boxes = regression.shifts_to_boxes(pred_shifts, inputs.anchor_points, inputs.images)
            pool_boxes = regression.filter_boxes_on_size(batched_boxes)

            L_reg = reg_loss(y=inputs.gt_deltas, y_hat=pred_shifts, inside_box_flags=inputs.point_labels,
                             scope=regression.scope, weight_pos=P.box_reg_pos_cls_weight, weight_neg=P.box_reg_neg_cls_weight,
                             batch_size=P.batch_size)
            keep_my_loss(L_reg)

        else:
            # For segmentation based we use gt boxes as our "predictions"
            pool_boxes = inputs.gt_boxes

        self.pool_boxes = pool_boxes

        # (5) PHOCS
        if build_phocs:
            WordModel = get_word_embedding_model(P.embed_model)
            word_embed = WordModel(scope='phoc', output_shape=(3, 9), exp_dir=experiment_dir, args=P)
            models.add_model(word_embed)
            pooled_phoc = word_embed.base_pooling(features, pool_boxes)
            phoc_logits = word_embed.phocs(pooled_phoc)
            self.pred_phocs = pred_phocs = tf.sigmoid(phoc_logits, name='phocs')

            # Use GT loss in segmentation based scenario - segmentation free uses random training (see below)
            if not segmentation_free:
                L_phoc = phoc_loss_func(y=inputs.gt_phocs[:, 1:], y_hat=phoc_logits, scope='phoc_loss', reduction=reduction)
                keep_my_loss(L_phoc)
        else:
            L_phoc = tf.constant(0.0, name='no_phocs')

        # (6) IoU Classifier
        if segmentation_free:
            iou_estimator = IoUPrediction(output_shape=(16, 64), args=P, exp_dir=experiment_dir, scope='iou')
            pooling_hmap = smooth_hmap if P.unet_size > 0 else hmap
            pooled_features = iou_estimator.base_pooling(pooling_hmap, pool_boxes) if not P.hmap_ablation else iou_estimator.base_pooling(input_img, pool_boxes)

            models.add_model(iou_estimator)
            pred_iou_logits = iou_estimator.iou(pooled_features)

            # Our predicted boxes (after filtering etc..)
            self.good_boxes = iou_estimator.get_good_boxes(pred_iou_logits, pool_boxes)

            if build_phocs:
                # Our predicted word embeddings (after filtering etc..)
                self.good_phocs = iou_estimator.get_good_embedding(pred_phocs)
        else:
            self.good_boxes = pool_boxes
            if build_phocs:
                self.good_phocs = pred_phocs

        # (7) Random Training for IoU and Phocs
        if segmentation_free:
            rnd_boxes, rnd_iou_labels, rnd_phocs = random_boxes_ops(inputs.gt_boxes, scope='roi_pool',
                                                                    num_classes=P.box_filter_num_clsses,
                                                                    default_image_size=P.target_size,
                                                                    gt_phocs=inputs.gt_phocs if build_phocs else None,
                                                                    phoc_dim=P.phoc_dim,
                                                                    iou_cls_lower_bound=P.iou_cls_lower_bound,
                                                                    boxes_per_class=P.boxes_per_class,
                                                                    batch_size=P.batch_size)

            self.rnd_phocs = rnd_phocs
            self.rnd_boxes = rnd_boxes
            self.rnd_iou_labels = rnd_iou_labels

            # (7a) Random IoU prediction
            if P.hmap_ablation:
                rnd_pooled_features = iou_estimator.base_pooling(input_img, rnd_boxes)
            else:
                rnd_pooled_features = iou_estimator.base_pooling(pooling_hmap, rnd_boxes)
            rnd_iou_logits = iou_estimator.iou(rnd_pooled_features)

            # (7b)Iou Loss
            L_iou = iou_loss(y=rnd_iou_labels, y_hat=rnd_iou_logits, scope='roi_pool')
            keep_my_loss(L_iou)

            # (7c) Random PHOC prediction
            if build_phocs:
                pooled_phoc = word_embed.base_pooling(features, rnd_boxes)
                v_phoc = pooled_phoc
                rnd_phoc_logits = word_embed.phocs(v_phoc)
                if P.aux_iou:
                    # Use IoU loss as regularizer for PHOC loss
                    aux_iou_logits = word_embed.aux_iou(v_phoc)
                    L_aux_iou = iou_loss(y=rnd_iou_labels, y_hat=aux_iou_logits, scope='aux_iou_loss')
                    keep_my_loss(L_aux_iou)

                # (7d) PHOC loss
                # We train phocs only on highly accuracte random boxes
                accuracte_phocs = tf.where(rnd_iou_labels >= P.min_iou_cls_for_phoc, name='acc_phocs_idx')[:, 0]
                good_iou_phocs = tf.gather(rnd_phocs, accuracte_phocs, name='acc_phocs')
                good_iou_logits = tf.gather(rnd_phoc_logits, accuracte_phocs, name='acc_phocs_logits')

                L_phoc = phoc_loss_func(y=good_iou_phocs, y_hat=good_iou_logits, scope='phoc_loss', reduction=reduction)
                keep_my_loss(L_phoc)

        self.update_ops = update_ops = get_update_ops()

        L2_regularization = tf.add_n(tf.losses.get_regularization_losses(), name='l2_reg') \
            if tf.losses.get_regularization_losses() else tf.constant(0., name='l2_reg')
        keep_my_loss(L2_regularization)

        # We can train parts of the network by setting train_vars to not None values (see command line help for accepted format)
        if P.train_vars is not None and (P.train_hmap or P.unet_size < 1):
            var_list = []
            loss_list = [L2_regularization]
            if 'fmap' in P.train_vars:
                var_list += feature_map.vars()
                loss_list += [P.heatmap_total_loss_weight * L_hmap]
            if 'hmap' in P.train_vars:
                var_list += heatmap.vars()
                if not 'fmap' in P.train_vars:
                    loss_list += [P.heatmap_total_loss_weight * L_hmap]
            if 'smoother' in P.train_vars and P.unet_size > 0:
                var_list += heatmap.vars()
                loss_list += [P.heatmap_total_loss_weight * L_smooth]
            if 'phoc' in P.train_vars and build_phocs:
                var_list += word_embed.vars()
                loss_list += [P.phoc_loss_weight * L_phoc]
                if P.aux_iou:
                    loss_list += [P.iou_predictions_loss_weight*L_aux_iou]
        else:
            var_list = feature_map.vars() + heatmap.vars() + hmap_smoother.vars()
            loss_list = [P.heatmap_total_loss_weight * L_hmap, P.heatmap_total_loss_weight * L_smooth,
                         P.phoc_loss_weight * L_phoc, L2_regularization]
            if build_phocs:
                var_list += word_embed.vars()

        L_hmap_phoc = tf.add_n(loss_list, name='hmap_and_phoc_loss')

        # Make train-op for hmap training
        if var_list:
            self.train_hmap, self.gs_hmap = get_train_op(L_hmap_phoc, P.lr_hmap, P, P.iters, var_list, update_ops=update_ops, name='hmap_train',
                                                         optimizer=tf.train.AdamOptimizer, beta1=0.5)
            models.add_vars(self.gs_hmap)

        if segmentation_free:
            if P.train_vars is not None and P.train_regression:
                box_var_list = []
                if 'reg' in P.train_vars:
                    box_var_list += regression.vars()
                if 'iou' in P.train_vars:
                    box_var_list += iou_estimator.vars()
            else:
                box_var_list = regression.vars() + iou_estimator.vars()
            L2_boxes_regularization = tf.add_n(tf.losses.get_regularization_losses('.*%s|%s.*$' % (regression.scope, iou_estimator.scope)), name='l2_box_reg')
            L_boxes = tf.add_n([L_reg, P.iou_predictions_loss_weight * L_iou, L2_boxes_regularization], name='reg_iou_loss')

            # Make train-op for regression training
            if box_var_list:
                self.train_boxes, self.gs_reg = get_train_op(L_boxes, P.lr_boxes, P, P.iters, box_var_list, update_ops=update_ops, name='boxes_train',
                                                   optimizer=tf.train.AdamOptimizer, beta1=0.5)
                models.add_vars(self.gs_reg)
            keep_my_loss(L_boxes)

        # Train summaries
        scalar_sums = my_losses()
        if var_list:
            scalar_sums += [self.gs_hmap]
        if segmentation_free:
            scalar_sums += [self.gs_reg]
        images_sum = [inputs.images, inputs.box_viz_images]
        images_sum += [hmap, inputs.gt_heatmap]
        if P.unet_size > 0:
            images_sum += [smooth_hmap]
        histo_sums = [hmap, inputs.gt_heatmap, features]
        if build_phocs:
            histo_sums += [inputs.gt_phocs]

        settings.make_summaries(scalars=scalar_sums, histograms=histo_sums, images=images_sum)
