from data.simple_pipe import PipelineBase, InBatchKeys
from data.data_extenders import phoc_embedding, from_image_to_heatmap, regression_bbox_targets, tf_boxes
from data.augmentations import Resize, ImageEmbed, GaussianNoise, KeepWordsOnly, DialationErosio, Slant, BoxRearange, PartilPage


def get_pipeline(params, train_iterator, num_producer_threads, augmentations=True, crop_words=False):
    # Params for easy access
    FMAP_W = params.feature_map_width
    FMAP_H = params.feature_map_height
    target_size = params.target_size
    batch_size = params.batch_size
    trim = params.gt_heatmap_trim_ratio

    # Data loader
    pipeline = PipelineBase(train_iterator, batch_size=batch_size, target_x=target_size[0], target_y=target_size[1])

    if crop_words:
        # Crop document to include only the region around word annotations
        pipeline.add_augmentation(KeepWordsOnly)

    if params.gray_augment and augmentations:
        # Apply gray scale augmentations
        pipeline.add_augmentation(DialationErosio, apply_prob=0.2)
        pipeline.add_augmentation(Slant, apply_prob=0.2)

    if params.gray_augment:
        # Transform input image into gray scale, without augmentation
        pipeline.add_augmentation(DialationErosio, apply_prob=0.0)

    # Augmentations before image resize
    if augmentations:
        pipeline.add_augmentation(PartilPage, apply_prob=params.partial_image_prob)

    pipeline.add_augmentation(Resize)

    # Augmentations on resized image
    if augmentations:
        pipeline.add_augmentation(GaussianNoise, prob=params.gaussian_noise_apply_prob)
        pipeline.add_augmentation(ImageEmbed, ratio_bounds=params.image_embedding_resize_ratio_bounds,
                                  prob=params.image_embedding_apply_prob)

    # Heatmap
    pipeline.add_extender('heatmap', from_image_to_heatmap, in_batch=InBatchKeys.vstack, trim=trim, with_border=params.hmap_border)
    # Regression
    pipeline.add_extender(('reg_target', 'reg_flags'), regression_bbox_targets, in_batch=InBatchKeys.vstack, fmap_w=FMAP_W,
                          fmap_h=FMAP_H)
    # TF Boxes
    pipeline.add_extender(('tf_gt_boxes', ), tf_boxes, in_batch=InBatchKeys.hstack)
    # Phoc Extenders
    pipeline.add_extender(('phocs', 'tf_gt_boxes'), phoc_embedding, in_batch=InBatchKeys.hstack)
    pipeline.run(num_producers=num_producer_threads)
    return pipeline
