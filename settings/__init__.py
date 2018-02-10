from collections import namedtuple
import tensorflow as tf
from tensorflow.contrib.keras.python.keras import backend as K
from input_stream import get_dataset_loader, log_params, write_params_to_args, get_dataset_loader_by_name
from lib import helpers as utils
from models.specs import word_embeddings, TFNetwork
from tensorflow.python import debug as tf_debug
from pipeline import get_pipeline

placeholders = namedtuple('X', ['images', 'box_viz_images', 'gt_heatmap', 'gt_boxes', 'gt_phocs', 'anchor_points', 'gt_deltas', 'point_labels', 'train_flag'])


def get_word_embedding_model(name):
    """ Helper to select a word embedding model from command line argument"""
    assert hasattr(word_embeddings, name), '%s does not exists' % name
    ModelCls = getattr(word_embeddings, name)
    assert issubclass(ModelCls, TFNetwork), '%s is not a Valid Word Embedding model' % name
    return ModelCls


def get_inputs(batch_size=1, target_size=(900, 1200), fmap=(112, 150), phoc_dim=0):
    """
    Get Model Inputs
    Generate graph placeholders and return them as a namedtuple
    """

    image = tf.placeholder(tf.float32, [batch_size, target_size[1], target_size[0], 3], name='image')

    box_viz = tf.placeholder(tf.float32, [batch_size, target_size[1], target_size[0], 3], name='box_viz_image')

    heatmap = tf.placeholder(tf.float32, [batch_size, target_size[1], target_size[0], 1], name='heatmap')

    # First coord is batch index
    tf_gt_boxes = tf.placeholder(tf.float32, [None, 5], name='gt_boxes')
    gt_phoc_tensor = tf.placeholder(tf.float32, [None, phoc_dim + 1], name='gt_phocs')

    relative_points = tf.placeholder(tf.float32, [1, fmap[0]*fmap[1], 2], name='relative_points')

    cntr_box_targets = tf.placeholder(tf.float32, [None, fmap[0]*fmap[1], 4], name='cntr_box_target')

    cntr_box_labels = tf.placeholder(tf.float32, [None, fmap[0]*fmap[1]], name='cntr_box_labels')

    # Use Keras' train mode placeholder
    is_training = K.learning_phase()

    return placeholders(image, box_viz, heatmap, tf_gt_boxes, gt_phoc_tensor, relative_points, cntr_box_targets, cntr_box_labels, is_training)


def feed_dict_from_dict(inputs, batch, pipe, params, train_mode=True):
    assert isinstance(inputs, placeholders), 'Pass Placeholders namedtuple'
    assert isinstance(batch, dict), 'Pass dict Batch'

    feed_dict = {inputs.images: batch['image'],
                 inputs.gt_heatmap: batch['heatmap'],
                 inputs.gt_boxes: batch['tf_gt_boxes'],
                 inputs.gt_deltas: batch['reg_target'] * params.regression_target_scaling_factor,
                 inputs.point_labels: batch['reg_flags'],
                 inputs.anchor_points: pipe.get_relative_points(fmap_w=params.feature_map_width, fmap_h=params.feature_map_height, as_batch=True),
                 inputs.train_flag: train_mode}

    if params.phoc_dim > 0:
        phocs = batch.get('phocs', None)
        if phocs is None:
            raise ValueError('Warning: phoc-dim is %d. While there no PHOCS availabale. Set phoc-dim to 0' % params.phoc_dim)
        else:
            feed_dict[inputs.gt_phocs] = batch['phocs']
    return feed_dict


def get_exp_dir_and_logger(experiment_dir):
    experiment_dir = utils.get_or_create_path(path_string=experiment_dir)
    logger = utils.Logger(log_dir=experiment_dir)
    return experiment_dir, logger


def get_session(P):
    """
    Produce TensorFlow session form arguments
    P needs to have the following:
        gpu_alloc - ratio of GPU memory to allocate
        tf_debug - boolean. If true run session with TF debugger

    """
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=P.gpu_alloc, allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    if P.tf_debug:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    return sess


def get_train_op(loss, lr, params, train_iters=None, var_list=None, update_ops=None, global_step=None, name='box_filter',
                 optimizer=tf.train.AdamOptimizer, **kwargs):
    """
    Generate Train op and global step for a loss
    """
    if global_step is None:
        global_step = tf.get_variable(name='%s_global_step' % name, dtype=tf.int32, initializer=0, trainable=False)
    train_iters = params.iters if train_iters is None else train_iters
    decay_steps = int(
        train_iters * params.decay_after_it_ratio) if params.decay_after_it_ratio < 1 else params.decay_after_it_ratio
    print('Decay after: %d' % decay_steps)
    lr = tf.train.exponential_decay(learning_rate=lr, global_step=global_step, decay_steps=decay_steps,
                                    decay_rate=params.learning_rate_decay, staircase=True)
    tf.add_to_collection('LearningRate', lr)

    opt = optimizer(learning_rate=lr, **kwargs)

    if update_ops is not None:
        with tf.control_dependencies(update_ops):
            train_op = opt.minimize(loss, var_list=var_list, global_step=global_step)
    else:
        train_op = opt.minimize(loss, var_list=var_list,
                                global_step=global_step)  # var_list=model.box_filter_model.model_vars,

    return train_op, global_step


def make_summaries(scalars=None, histograms=None, images=None, max_outputs=10):
    """ Make summary ops"""
    if scalars:
        assert isinstance(scalars, (list, tuple,))
        for s in scalars:
            tf.summary.scalar('scalar/%s' % s.name, s)

    if histograms:
        assert isinstance(histograms, (list, tuple,))
        for his in histograms:
            tf.summary.histogram('hist/%s' % his.name, his)

    if images:
        assert isinstance(images, (list, tuple,))
        for img in images:
            tf.summary.image('img/%s' % img.name, img, max_outputs=max_outputs)
    return
