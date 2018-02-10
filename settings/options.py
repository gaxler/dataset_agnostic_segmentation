from argparse import ArgumentParser


class Options(object):

    def __init__(self):
        p = self.parser = ArgumentParser(description=__doc__)

        p.add_argument('--batch_size', type=int, default=1, help='Number of images per batch')
        p.add_argument('--box-filter-num-clsses', '-filter-classes', type=int, default=5, help='Number of classes for IoU filtering')
        p.add_argument('--box-filter-prob-threshold', type=float, default=0.3, help='Min. word probability for word filtering')
        p.add_argument('--boxes-per-class', type=str, default=20, help='Box Filtering: Number of random boxes to generate per IoU class')
        p.add_argument('--dataset', type=str, required=True)
        p.add_argument('--decay-after-it-ratio', '-decay', type=float, default=50000, help='start learning rate decay aftex X% of iterations')
        p.add_argument('--learning-rate-decay', '-dec-rate', type=float, default=0.1, help='LR*X^(decay_after_it_ratio/iters')
        p.add_argument('--dropout', default=0.0, type=float, help='Dropout prob')
        p.add_argument('--data_type', required=True, type=str, help='What data set to use (separated by # to combine several). ')
        p.add_argument('--data_type_prob', default=None, type=str, help='Mix probabilities for several datasets')
        p.add_argument('--eval-run', action='store_true', help='Eval run evaluates the model with forward runs only and without augmentation')
        p.add_argument('--output-all', action='store_true', help='Output all predictions')
        p.add_argument('--experiment_dir', default='experiments', type=str)
        p.add_argument('--feature_map_height', '-fmaph', type=int, default=150, help='Height of feature map after image size reduced by convolution')
        p.add_argument('--feature_map_width', '-fmapw', type=int, default=112, help='Width of feature map after image size reduced by convolution')
        p.add_argument('--gpu_id', default=0, type=int, help="Set env var CUDA_VISIBLE_DEVICES to the provided number")
        p.add_argument('--gpu-alloc', default=0.99, type=float, help="Set GPU memory alocation ratio (default 0.9 (99% of mem)")
        p.add_argument('--gaussian-noise-apply-prob', default=0.0, type=float, help="Gaussian noise augmentation prob")
        p.add_argument('--hmap-ablation', action='store_true', help='Use original image as regression input (for abalation analysis')
        p.add_argument('--heatmap-pos-cls-weight', default=0.67, type=float, help='Hmap loss weighting - positive class')
        p.add_argument('--heatmap-neg-cls-weight', default=0.33, type=float, help='Hmap loss weighting - negative class')

        p.add_argument('--heatmap-L2-reg', '-l2-hmap', default=0.01, type=float, help='L2 regularization weigth for heatmap netwrok')
        p.add_argument('--heatmap-total-loss-weight', default=50, type=float, help='Weighting of heatmap loss in Heatmap Training total loss')
        p.add_argument('--box-reg-L2-reg', '-l2-reg', default=0.01, type=float, help='L2 regularization weigth for Regression Network')
        p.add_argument('--box-reg-pos-cls-weight', default=100., type=float, help='Labels highly unbalanced. Positive boxes get much higher weight than negative')
        p.add_argument('--box-reg-neg-cls-weight', default=1., type=float, help='Labels highly unbalanced. Negative boxes get much lower weight than positive')
        p.add_argument('--box-filter-L2-reg', '-l2-iou', default=0.01, type=float, help='L2 regularization weigth for IoU prediction classifier')
        p.add_argument('--iters', default=0, type=int, help='Number of iterations')
        p.add_argument('--image-embedding-apply-prob', default=0.5, type=int, help='Image Embedding noise augmentation prob')
        p.add_argument('--image-embedding-resize-ratio-bounds', default='0.75x1.0', type=str, help='Image Embedding resize bounds')
        p.add_argument('--image-normalize-const', default=255., type=float, help='Denominator for image normalization')
        p.add_argument('--iou-predictions-loss-weight', default=5., type=float, help='Weighting of IoU loss in Boxes Training total loss')
        p.add_argument('--iou-threshold-for-detection', default=0.6, type=float, help='IoU thresh with gt for true detection (Binary image defaults to 0.9)')
        p.add_argument('--iou-cls-lower-bound', default=None, type=float, help='IoU threshold to be considered positive (containing a word) box proposal')
        p.add_argument('--log-prefix', default=None, type=str, help='prefix to TensorBoard log dir')
        p.add_argument('--log-steps', default=500, type=int)
        p.add_argument('--name', type=str)
        p.add_argument('--min-iou-cls-for-phoc', '-iou-phoc', type=int, default=2, help='IoU class to qualify as good phoc')
        p.add_argument('--nms-overlap-thresh', '-nms', type=float, default=0.1, help='IoU threshold form nms')
        p.add_argument('--reset-gs', action='store_true', help="reset global_step")
        p.add_argument('--partial-image-prob', default=0.0, type=float, help='')
        p.add_argument('--num-of-classes-to-sum', default=3, type=int, help='When filtering boxes, how many classes to sum to get class score')
        p.add_argument('--phoc-dim', default=540, type=int, help='Size of PHOC vector')
        p.add_argument('--phoc-loss-weight', default=150, type=float, help='Weighting of PHOC loss in Heatmap Training total loss')
        p.add_argument('--regression-target-scaling-factor',  default=1500, type=float,
                       help='Scale for regression output. Defaults to 1500 to allow sensitivity ~single pixel (large side is 1200 by default)')
        p.add_argument('--save-steps', type=int, default=2000)
        p.add_argument('--segment-free', action='store_false')
        p.add_argument('--seed', default=128, type=int)
        p.add_argument('--score-thresh', default=0.5, type=float, help='Filter score to filter with')
        p.add_argument('--stat-prefix', type=str, default=None, help='Directory to keep the stats in')
        p.add_argument('--gt_heatmap_trim_ratio', '-trim', default=0.2, type=float, help="Bounding boxes will be shrinked by this ratio when creating ground truth heatmap")

        p.add_argument('--tf-debug', action='store_true', help='Run TF debbuger (command line only')
        p.add_argument('--target-size', type=str, default='900x1200', help='Input image size in pixels WIDTHxHEIGHT')
        p.add_argument('--hmap-border', action='store_true', help='Add border class')

    @staticmethod
    def split_arg(arg, type, on='x'):
        arg = str(arg)
        return tuple(map(type, arg.split(on)))

    def conflict_assertions(self, args):
        return args

    def parse(self):
        args = self.parser.parse_args()
        # target size to tuple
        ts = args.target_size
        args.target_size = Options.split_arg(ts, int)
        im_emb = args.image_embedding_resize_ratio_bounds
        args.image_embedding_resize_ratio_bounds = Options.split_arg(im_emb, float)

        args = self.conflict_assertions(args)
        return args


class BoxSegmentWithPHOCOptions(Options):
    def __init__(self):
        super(BoxSegmentWithPHOCOptions, self).__init__()
        p = self.parser

        p.add_argument('--aux-iou', action='store_true', help='use IoU loss as a regularizer for PHOC')
        p.add_argument('--crop-words', action='store_true', help='Continue train = load vars in checkpoint')
        p.add_argument('--gray-augment', action='store_true', help='Dialation augment + gray scale image')
        p.add_argument('--embed-model', default='MyOldPHOC', help='Embedding model class to use')
        p.add_argument('--load-vars', type=str, default=None, help='Load specific vars manually')
        p.add_argument('--lr-hmap', type=float, default=0.0001, help='Learning rate for hmap train (hmap + phocs + [reconstruction in pretrain])')
        p.add_argument('--lr-boxes', type=float, default=0.0005, help='LR regression + IoU classification')
        p.add_argument('--phoc-act', type=str, default='relu', help='PHOC net activation function')
        p.add_argument('--unet-size', type=int, default=8)
        p.add_argument('--unet-depth', type=int, default=5)
        p.add_argument('--unet-with-img', action='store_true', help='Use input image as input to U-NET')
        p.add_argument('--train-regression', action='store_true')
        p.add_argument('--train-hmap', action='store_true')
        p.add_argument('--tiny-phoc', action='store_true')
        p.add_argument('--bigger-phoc', action='store_true')
        p.add_argument('--std-phoc', action='store_true')
        p.add_argument('--train-vars', default=None, type=str, help='Which vars to train. separate by \'x\' '
                                                                      'values can be {fmap, hmap, smoother, phoc, reg, iou}. e,g fmapxhmap => train only feature and heatmap')
        p.add_argument('--train-prcnt', default=None, type=int, help='Precnt of training data. allowrd [10, 30, 70]')

    def parse(self):
        args = super(BoxSegmentWithPHOCOptions, self).parse()
        if args.train_vars is not None:
            args.train_vars = BoxSegmentWithPHOCOptions.split_arg(args.train_vars, str)
        return args


class EvalOptions(object):

    def __init__(self):
        p = self.parser = ArgumentParser(description=__doc__)

        p.add_argument('--stat-prefix', '-stat', default='eval', type=str, help='Directory to look for stats')
        p.add_argument('--eval-protocol', required=True, help='what evaluation should we do? [mAP, segment, struct_segment]')
        p.add_argument('--experiment_dir', default='experiments', type=str)
        p.add_argument('--eval-dir', default=None, type=str, help='Specific Dir to evaluate')
        p.add_argument('--dataset', type=str, default=None, help='Dataset for specific dir')
        p.add_argument('--data-type', type=str, default='val1', help='type of dataset to use {train, val1, val2}')
        p.add_argument('--use-test-data',  action='store_true', help='Flag to use actual test data')
        p.add_argument('--max-words', type=int, default=None, help='Maximum words to query (For estimated results)')
        p.add_argument('--iou', type=float, default=0.9, help='IoU threshold for segmentation eval (defaults to 0.9 for ICDAR')

    def parse(self):
        args = self.parser.parse_args()
        return args
