from __future__ import print_function
from pathlib2 import Path
import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets import resnet_v2

from tensorflow.contrib import slim

resnet_v2_block = resnet_v2.resnet_v2_block


def get_vars(reg_ex=None):
    v = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, reg_ex)
    return v

def get_update_ops(reg_ex=None):
    v = tf.get_collection(tf.GraphKeys.UPDATE_OPS, reg_ex)
    return v

def keep_my_loss(v):
    if isinstance(v, (tuple, list)):
        for _v in v:
            tf.add_to_collection('my_losses', _v)
    else:
        tf.add_to_collection('my_losses', v)


def my_losses(reg_ex=None):
    return tf.get_collection('my_losses', reg_ex)


def resnet_from_blocks(inputs, blocks, num_classes=None, is_training=None, global_pool=False, output_stride=None, reuse=None, scope=None):
    return resnet_v2.resnet_v2(inputs, blocks, num_classes, is_training, global_pool, output_stride, include_root_block=True, reuse=reuse, scope=scope)


class ModelsSaveLoadManager(object):
    """
    Helper to manage weights saving and loading of different parts of the model
    """

    def __init__(self, exp_dir):
        self._models = []
        self._additional_vars = []

        self._saver = None
        self.exp_dir = exp_dir

    def add_model(self, model):
        assert isinstance(model, TFNetwork), 'Model must be a %s class' % TFNetwork.__name__
        self._models.append(model)

    def save(self, sess, global_step=None):
        for model in self._models:
            model.save(sess, global_step)
        self._save(sess, global_step)

    def load(self, sess):
        for model in self._models:
            model.load(sess)
        self._load(sess)

    def vars(self):
        return self._additional_vars

    def add_vars(self, var_lst):
        if isinstance(var_lst, (list, tuple,)):
            for v in var_lst:
                self._additional_vars.append(v)
        else:
            self._additional_vars.append(var_lst)

    def get_saver(self):
        if self._saver is None:
            self._saver = tf.train.Saver(max_to_keep=2, var_list=self.vars())
        return self._saver

    def _save(self, sess, global_step=None):
        if self._additional_vars:
            model_saver = self.get_saver()
            if global_step is not None:
                return model_saver.save(sess, save_path=str(self.exp_dir / ('global_vars')), global_step=global_step,
                                        latest_filename='global_vars_ckpt')
            return model_saver.save(sess, save_path=str(self.exp_dir / ('global_vars')),  latest_filename='global_vars_ckpt')

    def _load(self, sess):
        if self._additional_vars:
            model_saver = self.get_saver()
            ckpt = tf.train.latest_checkpoint(str(self.exp_dir), latest_filename='global_vars_ckpt')
            if ckpt is None:
                print('No ckpt found...')
                return
            print('Loading %s' % str(ckpt))
            model_saver.restore(sess, ckpt)
        return


class TFNetwork(object):

    def __init__(self, exp_dir, args, scope, **kwargs):
        self.exp_dir = exp_dir
        self.scope = scope
        self.args = args
        self._saver = None

    def vars(self):
        return get_vars('.*%s.*$' % self.scope)

    def get_saver(self):
        if self._saver is None:
            self._saver = tf.train.Saver(max_to_keep=2, var_list=self.vars(), name='%s_saver' % self.scope)#, ignore_missing_variables=True)
        return self._saver

    def save(self, sess, global_step=None):
        model_saver = self.get_saver()
        if global_step is not None:
            return model_saver.save(sess, save_path=str(self.exp_dir / ('%s_model' % self.scope)), global_step=global_step,
                                    latest_filename='%s_ckpt' % self.scope)
        return model_saver.save(sess, save_path=str(self.exp_dir / ('%s_model' % self.scope)), latest_filename='%s_ckpt' % self.scope)

    def load(self, sess):
        model_saver = self.get_saver()
        ckpt = tf.train.latest_checkpoint(str(self.exp_dir), latest_filename='%s_ckpt' % self.scope)
        if ckpt is None:
            print('[ %s ] No ckpt found...' % self.scope)
            return
        print('Loading %s' % str(ckpt))
        init_op, init_feed = slim.assign_from_checkpoint(model_path=ckpt, var_list=self.vars(), ignore_missing_vars=True)
        sess.run(init_op, init_feed)
        # model_saver.restore(sess, ckpt)
        return

    def get_reuse(self, counter):
        if counter > 0:
            return True
        return None


# class FeaturGANVarSaver(object):
#     _fmap = '.*fmap/.*$'
#     _hmap = '.*hmap/.*$'
#     _reconstruction = '.*rec/.*$'
#     _regression = '.*box_reg/.*$'
#     _iou = '.*iou_pred/.*$'
#     _iou_pool = '.*iou_pool/.*$'
#     _phocs = '.*phocs/.*$'
#     _phocs_pool = '.*phoc_pool/.*$'
#     _discriminator = '.*disc/.*$'
#     _global_step = '.*_global_step.*$'
#
#     # Unsupervised helpers
#     _smoother = '.*smoother.*$'
#     _domain_orthogonality = '.*domain_orthogonality.*$'
#
#     def __init__(self, real, fake, base_dir, logger=None):
#         self.base_dir = Path(base_dir).parent
#         self.baseline = 'baseline'
#         self.pretrain = 'pretrain'
#         self.unsup_boxes = 'unsup_boxes'
#         self.unsup_phoc = 'unsup_phoc'
#         self.Dtrain = 'd_train'
#         self.Gtrain = 'g_train'
#         self._exp_dir = '%s_%s' % (real, fake)
#
#         self._op_and_feed = []
#         self._save_vars = []
#
#         self.logger = print if logger is None else logger
#
#         self.saver = None
#
#     def _path(self, stage, name=None):
#         if name is not None:
#             return self.base_dir / name
#         return self.base_dir / ('%s_%s' % (self._exp_dir, stage))
#
#     def save_path(self, stage, eval, name=None):
#         suffix = 'train'
#         if eval:
#             suffix = 'eval'
#         return self._path(stage, name=name) / suffix
#
#     def fmap(self):
#         return self._vars(self._fmap)
#
#     def hmap(self):
#         return self._vars(self._hmap)
#
#     def smoother(self):
#         return self._vars(self._smoother)
#
#     def orthogonality(self):
#         return self._vars(self._domain_orthogonality)
#
#     def reconstruction(self):
#         return self._vars(self._reconstruction)
#
#     def regression(self):
#         return self._vars(self._regression)
#
#     def iou(self):
#         return self._vars([self._iou_pool, self._iou])
#
#     def phocs(self):
#         return self._vars([self._phocs_pool, self._phocs])
#
#     def discriminator(self):
#         return self._vars(self._discriminator)
#
#     def global_step(self):
#         return self._vars(self._global_step)
#
#     def save(self, sess, stage, global_step, name=None, train_type=None):
#         if not self._save_vars:
#             self._save_vars = tf.global_variables()
#         if self.saver is None:
#             var_list = self._save_vars + self.global_step()
#             self.saver = tf.train.Saver(max_to_keep=2, var_list=var_list)
#         if train_type is None:
#             self.saver.save(sess, save_path=str(self._path(stage, name=name) / 'model'), global_step=global_step)
#         else:
#             self.saver.save(sess, save_path=str(self._path(stage, name=name) / ('%s_model' % train_type)), global_step=global_step)
#
#     def load(self, sess):
#         for op, feed, path in self._op_and_feed:
#             self.logger('##### Loading Variables ######')
#             for x in feed.keys():
#                 self.logger('%s: %s (%s)' % (x.name, str(x.shape.as_list()), path))
#             sess.run(op, feed)
#
#     def vars_for(self, _from, global_step=True, name=None, load=True):
#         options = {self.baseline: self._vars([self._fmap, self._hmap, self._regression, self._iou, self._iou_pool, self._phocs, self._phocs_pool]),
#                    self.unsup_boxes: self._vars([self._fmap, self._hmap, self._smoother, self._regression, self._iou, self._iou_pool, self._phocs, self._phocs_pool]),
#                    self.unsup_phoc: self._vars([self._fmap, self._hmap, self._regression, self._iou, self._iou_pool, self._phocs, self._phocs_pool, self._domain_orthogonality]),
#                    self.pretrain: self._vars([self._fmap, self._hmap, self._phocs, self._phocs_pool, self._reconstruction]),
#                    self.Dtrain: self._vars([self._fmap, self._discriminator]),
#                    self.Gtrain: self._vars([self._fmap, self._discriminator]),
#                    }
#         # ckpt = self.checkpoint('%s_%s' % (self._exp_dir, _from))
#         if load:
#             ckpt = self.checkpoint(self._path(_from, name=name), abs_loc=name is not None)
#             if ckpt is not None:
#                 self.logger('Adding load Ops from %s' % str(ckpt))
#                 var_list = options[_from] + self.global_step() if global_step else options[_from]
#                 self.init_op_and_feed(ckpt, var_list)
#             else:
#                 self.logger('No checkpoint for %s initializing' % (str(self._path(_from, name=name))))
#         added_vars = options[_from] + self.global_step() if global_step else options[_from]
#         self._save_vars += added_vars
#
#     def take_from(self, what, _from, global_step=True, name=None):
#         if isinstance(what, (list, tuple)):
#             var_list = [x for lst in what for x in lst]
#         elif isinstance(what, str):
#             what = what.split('#')[1:]
#             var_list = self._vars([getattr(self, '_%s' % x) for x in what])
#         else:
#             raise ValueError
#
#         var_list = var_list + self.global_step() if global_step else var_list
#         ckpt = self.checkpoint(self._path(_from, name=name), abs_loc=name is not None)
#         if ckpt is not None:
#             self.init_op_and_feed(ckpt, var_list)
#         else:
#             self.logger('No checkpoint for %s initializing' % _from)
#
#     def checkpoint(self, loc, abs_loc=False):
#         if abs_loc:
#             self.logger('Looking in: %s' % str(loc))
#             return tf.train.latest_checkpoint(str(loc))
#         # ckpt = str(self.base_dir / loc)
#         ckpt = str(loc)
#         self.logger('Looking in: %s' % str(ckpt))
#         return tf.train.latest_checkpoint(ckpt)
#
#     def init_op_and_feed(self, path, var_list):
#         init_op, init_feed = slim.assign_from_checkpoint(model_path=path, var_list=var_list, ignore_missing_vars=True)
#         self._op_and_feed.append((init_op, init_feed, path))
#
#     def _vars(self, regex):
#         if isinstance(regex, (list, tuple)):
#             v = []
#             for r in regex:
#                 v += get_vars(r)
#             return v
#         return get_vars(regex)
#



