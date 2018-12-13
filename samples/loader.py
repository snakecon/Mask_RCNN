from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import random
from distutils.version import LooseVersion

import numpy as np
import skimage.color
import skimage.io
import skimage.io
import skimage.transform
import tensorflow as tf

from mrcnn import visualize


class MaskrcnnCocoConfig(object):
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # COCO has 80 classes

    # NUMBER OF GPUs to use. When using only a CPU, this needs to be set to 1.
    GPU_COUNT = 1

    # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
    STEPS_PER_EPOCH = 1000

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 50

    # Backbone network architecture
    # Supported values are: resnet50, resnet101.
    # You can also provide a callable that should have the signature
    # of model.resnet_graph. If you do so, you need to supply a callable
    # to COMPUTE_BACKBONE_SHAPE as well
    BACKBONE = "resnet101"

    # Only useful if you supply a callable to BACKBONE. Should compute
    # the shape of each layer of the FPN Pyramid.
    # See model.compute_backbone_shapes
    COMPUTE_BACKBONE_SHAPE = None

    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    # Size of the fully-connected layers in the classification graph
    FPN_CLASSIF_FC_LAYERS_SIZE = 1024

    # Size of the top-down layers used to build the feature pyramid
    TOP_DOWN_PYRAMID_SIZE = 256

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]

    # Anchor stride
    # If 1 then anchors are created for each cell in the backbone feature map.
    # If 2, then anchors are created for every other cell, and so on.
    RPN_ANCHOR_STRIDE = 1

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256

    # ROIs kept after tf.nn.top_k and before non-maximum suppression
    PRE_NMS_LIMIT = 6000

    # ROIs kept after non-maximum suppression (training and inference)
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Input image resizing
    # Generally, use the "square" resizing mode for training and predicting
    # and it should work well in most cases. In this mode, images are scaled
    # up such that the small side is = IMAGE_MIN_DIM, but ensuring that the
    # scaling doesn't make the long side > IMAGE_MAX_DIM. Then the image is
    # padded with zeros to make it a square so multiple images can be put
    # in one batch.
    # Available resizing modes:
    # none:   No resizing or padding. Return the image unchanged.
    # square: Resize and pad with zeros to get a square image
    #         of size [max_dim, max_dim].
    # pad64:  Pads width and height with zeros to make them multiples of 64.
    #         If IMAGE_MIN_DIM or IMAGE_MIN_SCALE are not None, then it scales
    #         up before padding. IMAGE_MAX_DIM is ignored in this mode.
    #         The multiple of 64 is needed to ensure smooth scaling of feature
    #         maps up and down the 6 levels of the FPN pyramid (2**6=64).
    # crop:   Picks random crops from the image. First, scales the image based
    #         on IMAGE_MIN_DIM and IMAGE_MIN_SCALE, then picks a random crop of
    #         size IMAGE_MIN_DIM x IMAGE_MIN_DIM. Can be used in training only.
    #         IMAGE_MAX_DIM is not used in this mode.
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024
    # Minimum scaling ratio. Checked after MIN_IMAGE_DIM and can force further
    # up scaling. For example, if set to 2 then images are scaled up to double
    # the width and height, or more, even if MIN_IMAGE_DIM doesn't require it.
    # Howver, in 'square' mode, it can be overruled by IMAGE_MAX_DIM.
    IMAGE_MIN_SCALE = 0
    # Number of color channels per image. RGB = 3, grayscale = 1, RGB-D = 4
    # Changing this requires other changes in the code. See the WIKI for more
    # details: https://github.com/matterport/Mask_RCNN/wiki
    IMAGE_CHANNEL_COUNT = 3

    # Image mean (RGB)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 200

    # Percent of positive ROIs used to train classifier/mask heads
    ROI_POSITIVE_RATIO = 0.33

    # Pooled ROIs
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14

    # Shape of output mask
    # To change this you also need to change the neural network mask branch
    MASK_SHAPE = [28, 28]

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 100

    # Bounding box refinement standard deviation for RPN and final detections.
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 100

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.7

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimizer
    # implementation.
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001

    # Loss weights for more precise optimization.
    # Can be used for R-CNN training setup.
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }

    # Use RPN ROIs or externally generated ROIs for training
    # Keep this True for most situations. Set to False if you want to train
    # the head branches on ROI generated by code rather than the ROIs from
    # the RPN. For example, to debug the classifier head without having to
    # train the RPN.
    USE_RPN_ROIS = True

    # Train or freeze batch normalization layers
    #     None: Train BN layers. This is the normal mode
    #     False: Freeze BN layers. Good when using a small batch size
    #     True: (don't use). Set layer in training mode even when predicting
    TRAIN_BN = False  # Defaulting to False since batch size is often small

    # Gradient norm clipping
    GRADIENT_CLIP_NORM = 5.0

    def __init__(self):
        """Set values of computed attributes."""
        # Effective batch size
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

        # Input image size
        if self.IMAGE_RESIZE_MODE == "crop":
            self.IMAGE_SHAPE = np.array([self.IMAGE_MIN_DIM, self.IMAGE_MIN_DIM,
                                         self.IMAGE_CHANNEL_COUNT])
        else:
            self.IMAGE_SHAPE = np.array([self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM,
                                         self.IMAGE_CHANNEL_COUNT])

        # Image meta data length
        # See compose_image_meta() for details
        self.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")


class MaskrcnnModel(object):
    def __init__(self):
        self.config = MaskrcnnCocoConfig()
        self.config.display()


        # COCO Class names
        # Index of the class in the list is its ID. For example, to get ID of
        # the teddy bear class, use: class_names.index('teddy bear')
        self.class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                       'bus', 'train', 'truck', 'boat', 'traffic light',
                       'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                       'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                       'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                       'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                       'kite', 'baseball bat', 'baseball glove', 'skateboard',
                       'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                       'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                       'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                       'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                       'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                       'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                       'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                       'teddy bear', 'hair drier', 'toothbrush']

        self.graph = self._load_graph('../mask_rcnn_coco.pb')

        # Input.
        self.input_image_placeholder = self.graph.get_tensor_by_name("infer/input_image:0")
        self.input_image_meta_placeholder = self.graph.get_tensor_by_name(
            "infer/input_image_meta:0")
        self.input_anchors_placeholder = self.graph.get_tensor_by_name("infer/input_anchors:0")

        # Output.
        self.mrcnn_detection = self.graph.get_tensor_by_name("infer/mrcnn_detection/Reshape_1:0")
        self.mrcnn_class = self.graph.get_tensor_by_name("infer/mrcnn_class/Reshape_1:0")
        self.mrcnn_bbox = self.graph.get_tensor_by_name("infer/mrcnn_bbox/Reshape:0")
        self.mrcnn_mask = self.graph.get_tensor_by_name("infer/mrcnn_mask/Reshape_1:0")
        self.roi = self.graph.get_tensor_by_name("infer/ROI/packed_2:0")
        self.rpn_class = self.graph.get_tensor_by_name("infer/rpn_class/concat:0")
        self.rpn_bbox = self.graph.get_tensor_by_name("infer/rpn_bbox/concat:0")

        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(graph=self.graph, config=tf.ConfigProto(gpu_options=gpu_options))

    def inference(self, img_rgb):
        # Mold inputs to format expected by the neural network
        molded_images, image_metas, windows = self.mold_inputs([img_rgb])

        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images[0].shape

        # Anchors
        anchors = self.get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)

        # Run forward pass to calculate embeddings
        detections, mrcnn_class, mrcnn_bbox, mrcnn_mask, rpn_rois, rpn_class, rpn_bbox = self.sess.run(
            [self.mrcnn_detection, self.mrcnn_class, self.mrcnn_bbox, self.mrcnn_mask, self.roi,
             self.rpn_class, self.rpn_bbox], feed_dict={
                self.input_image_placeholder: molded_images,
                self.input_image_meta_placeholder: image_metas,
                self.input_anchors_placeholder: anchors,
            })

        # Process detections
        final_rois, final_class_ids, final_scores, final_masks = \
            self.unmold_detections(detections[0], mrcnn_mask[0],
                                   image.shape, molded_images[0].shape,
                                   windows[0])

        return final_rois, final_masks, final_class_ids, final_scores

    def _load_graph(self, frozen_graph_filename):
        with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(
                graph_def,
                input_map=None,
                return_elements=None,
                name="infer",
                op_dict=None,
                producer_op_list=None
            )

        return graph

    def unmold_detections(self, detections, mrcnn_mask, original_image_shape,
                          image_shape, window):
        """Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the
        application.

        detections: [N, (y1, x1, y2, x2, class_id, score)] in normalized coordinates
        mrcnn_mask: [N, height, width, num_classes]
        original_image_shape: [H, W, C] Original image shape before resizing
        image_shape: [H, W, C] Shape of the image after resizing and padding
        window: [y1, x1, y2, x2] Pixel coordinates of box in the image where the real
                image is excluding the padding.

        Returns:
        boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, num_instances] Instance masks
        """
        # How many detections do we have?
        # Detections array is padded with zeros. Find the first class_id == 0.
        zero_ix = np.where(detections[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]
        masks = mrcnn_mask[np.arange(N), :, :, class_ids]

        # Translate normalized coordinates in the resized image to pixel
        # coordinates in the original image before resizing
        window = self.norm_boxes(window, image_shape[:2])
        wy1, wx1, wy2, wx2 = window
        shift = np.array([wy1, wx1, wy1, wx1])
        wh = wy2 - wy1  # window height
        ww = wx2 - wx1  # window width
        scale = np.array([wh, ww, wh, ww])
        # Convert boxes to normalized coordinates on the window
        boxes = np.divide(boxes - shift, scale)
        # Convert boxes to pixel coordinates on the original image
        boxes = self.denorm_boxes(boxes, original_image_shape[:2])

        # Filter out detections with zero area. Happens in early training when
        # network weights are still random
        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            N = class_ids.shape[0]

        # Resize masks to original image size and set boundary threshold.
        full_masks = []
        for i in range(N):
            # Convert neural network mask to full size mask
            full_mask = self.unmold_mask(masks[i], boxes[i], original_image_shape)
            full_masks.append(full_mask)
        full_masks = np.stack(full_masks, axis=-1) \
            if full_masks else np.empty(original_image_shape[:2] + (0,))

        return boxes, class_ids, scores, full_masks

    def get_anchors(self, image_shape):
        """Returns anchor pyramid for the given image size."""
        backbone_shapes = self.compute_backbone_shapes(self.config, image_shape)
        # Cache anchors and reuse if image shape is the same
        if not hasattr(self, "_anchor_cache"):
            self._anchor_cache = {}
        if not tuple(image_shape) in self._anchor_cache:
            # Generate Anchors
            a = self.generate_pyramid_anchors(
                self.config.RPN_ANCHOR_SCALES,
                self.config.RPN_ANCHOR_RATIOS,
                backbone_shapes,
                self.config.BACKBONE_STRIDES,
                self.config.RPN_ANCHOR_STRIDE)
            # Keep a copy of the latest anchors in pixel coordinates because
            # it's used in inspect_model notebooks.
            # TODO: Remove this after the notebook are refactored to not use it
            self.anchors = a
            # Normalize coordinates
            self._anchor_cache[tuple(image_shape)] = self.norm_boxes(a, image_shape[:2])
        return self._anchor_cache[tuple(image_shape)]

    def mold_inputs(self, images):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        images: List of image matrices [height,width,depth]. Images can have
            different sizes.

        Returns 3 Numpy matrices:
        molded_images: [N, h, w, 3]. Images resized and normalized.
        image_metas: [N, length of meta data]. Details about each image.
        windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
            original image (padding excluded).
        """
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            # Resize image
            # TODO: move resizing to mold_image()
            molded_image, window, scale, padding, crop = self.resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                min_scale=self.config.IMAGE_MIN_SCALE,
                max_dim=self.config.IMAGE_MAX_DIM,
                mode=self.config.IMAGE_RESIZE_MODE)
            molded_image = self.mold_image(molded_image, self.config)
            # Build image_meta
            image_meta = self.compose_image_meta(
                0, image.shape, molded_image.shape, window, scale,
                np.zeros([self.config.NUM_CLASSES], dtype=np.int32))
            # Append
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        # Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

    def resize_image(self, image, min_dim=None, max_dim=None, min_scale=None, mode="square"):
        """Resizes an image keeping the aspect ratio unchanged.

        min_dim: if provided, resizes the image such that it's smaller
            dimension == min_dim
        max_dim: if provided, ensures that the image longest side doesn't
            exceed this value.
        min_scale: if provided, ensure that the image is scaled up by at least
            this percent even if min_dim doesn't require it.
        mode: Resizing mode.
            none: No resizing. Return the image unchanged.
            square: Resize and pad with zeros to get a square image
                of size [max_dim, max_dim].
            pad64: Pads width and height with zeros to make them multiples of 64.
                   If min_dim or min_scale are provided, it scales the image up
                   before padding. max_dim is ignored in this mode.
                   The multiple of 64 is needed to ensure smooth scaling of feature
                   maps up and down the 6 levels of the FPN pyramid (2**6=64).
            crop: Picks random crops from the image. First, scales the image based
                  on min_dim and min_scale, then picks a random crop of
                  size min_dim x min_dim. Can be used in training only.
                  max_dim is not used in this mode.

        Returns:
        image: the resized image
        window: (y1, x1, y2, x2). If max_dim is provided, padding might
            be inserted in the returned image. If so, this window is the
            coordinates of the image part of the full image (excluding
            the padding). The x2, y2 pixels are not included.
        scale: The scale factor used to resize the image
        padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
        """
        # Keep track of image dtype and return results in the same dtype
        image_dtype = image.dtype
        # Default window (y1, x1, y2, x2) and default scale == 1.
        h, w = image.shape[:2]
        window = (0, 0, h, w)
        scale = 1
        padding = [(0, 0), (0, 0), (0, 0)]
        crop = None

        if mode == "none":
            return image, window, scale, padding, crop

        # Scale?
        if min_dim:
            # Scale up but not down
            scale = max(1, min_dim / min(h, w))
        if min_scale and scale < min_scale:
            scale = min_scale

        # Does it exceed max dim?
        if max_dim and mode == "square":
            image_max = max(h, w)
            if round(image_max * scale) > max_dim:
                scale = max_dim / image_max

        # Resize image using bilinear interpolation
        if scale != 1:
            image = self.resize(image, (round(h * scale), round(w * scale)),
                                preserve_range=True)

        # Need padding or cropping?
        if mode == "square":
            # Get new height and width
            h, w = image.shape[:2]
            top_pad = (max_dim - h) // 2
            bottom_pad = max_dim - h - top_pad
            left_pad = (max_dim - w) // 2
            right_pad = max_dim - w - left_pad
            padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
            image = np.pad(image, padding, mode='constant', constant_values=0)
            window = (top_pad, left_pad, h + top_pad, w + left_pad)
        elif mode == "pad64":
            h, w = image.shape[:2]
            # Both sides must be divisible by 64
            assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
            # Height
            if h % 64 > 0:
                max_h = h - (h % 64) + 64
                top_pad = (max_h - h) // 2
                bottom_pad = max_h - h - top_pad
            else:
                top_pad = bottom_pad = 0
            # Width
            if w % 64 > 0:
                max_w = w - (w % 64) + 64
                left_pad = (max_w - w) // 2
                right_pad = max_w - w - left_pad
            else:
                left_pad = right_pad = 0
            padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
            image = np.pad(image, padding, mode='constant', constant_values=0)
            window = (top_pad, left_pad, h + top_pad, w + left_pad)
        elif mode == "crop":
            # Pick a random crop
            h, w = image.shape[:2]
            y = random.randint(0, (h - min_dim))
            x = random.randint(0, (w - min_dim))
            crop = (y, x, min_dim, min_dim)
            image = image[y:y + min_dim, x:x + min_dim]
            window = (0, 0, min_dim, min_dim)
        else:
            raise Exception("Mode {} not supported".format(mode))
        return image.astype(image_dtype), window, scale, padding, crop

    def unmold_mask(self, mask, bbox, image_shape):
        """Converts a mask generated by the neural network to a format similar
        to its original shape.
        mask: [height, width] of type float. A small, typically 28x28 mask.
        bbox: [y1, x1, y2, x2]. The box to fit the mask in.

        Returns a binary mask with the same size as the original image.
        """
        threshold = 0.5
        y1, x1, y2, x2 = bbox
        mask = self.resize(mask, (y2 - y1, x2 - x1))
        mask = np.where(mask >= threshold, 1, 0).astype(np.bool)

        # Put the mask in the right location.
        full_mask = np.zeros(image_shape[:2], dtype=np.bool)
        full_mask[y1:y2, x1:x2] = mask
        return full_mask

    def generate_anchors(self, scales, ratios, shape, feature_stride, anchor_stride):
        """
        scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
        ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
        shape: [height, width] spatial shape of the feature map over which
                to generate anchors.
        feature_stride: Stride of the feature map relative to the image in pixels.
        anchor_stride: Stride of anchors on the feature map. For example, if the
            value is 2 then generate anchors for every other feature map pixel.
        """
        # Get all combinations of scales and ratios
        scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
        scales = scales.flatten()
        ratios = ratios.flatten()

        # Enumerate heights and widths from scales and ratios
        heights = scales / np.sqrt(ratios)
        widths = scales * np.sqrt(ratios)

        # Enumerate shifts in feature space
        shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
        shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
        shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

        # Enumerate combinations of shifts, widths, and heights
        box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
        box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

        # Reshape to get a list of (y, x) and a list of (h, w)
        box_centers = np.stack(
            [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
        box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

        # Convert to corner coordinates (y1, x1, y2, x2)
        boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                                box_centers + 0.5 * box_sizes], axis=1)
        return boxes

    def generate_pyramid_anchors(self, scales, ratios, feature_shapes, feature_strides,
                                 anchor_stride):
        """Generate anchors at different levels of a feature pyramid. Each scale
        is associated with a level of the pyramid, but each ratio is used in
        all levels of the pyramid.

        Returns:
        anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
            with the same order of the given scales. So, anchors of scale[0] come
            first, then anchors of scale[1], and so on.
        """
        # Anchors
        # [anchor_count, (y1, x1, y2, x2)]
        anchors = []
        for i in range(len(scales)):
            anchors.append(self.generate_anchors(scales[i], ratios, feature_shapes[i],
                                                 feature_strides[i], anchor_stride))
        return np.concatenate(anchors, axis=0)

    def norm_boxes(self, boxes, shape):
        """Converts boxes from pixel coordinates to normalized coordinates.
        boxes: [N, (y1, x1, y2, x2)] in pixel coordinates
        shape: [..., (height, width)] in pixels

        Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
        coordinates it's inside the box.

        Returns:
            [N, (y1, x1, y2, x2)] in normalized coordinates
        """
        h, w = shape
        scale = np.array([h - 1, w - 1, h - 1, w - 1])
        shift = np.array([0, 0, 1, 1])
        return np.divide((boxes - shift), scale).astype(np.float32)

    def denorm_boxes(self, boxes, shape):
        """Converts boxes from normalized coordinates to pixel coordinates.
        boxes: [N, (y1, x1, y2, x2)] in normalized coordinates
        shape: [..., (height, width)] in pixels

        Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
        coordinates it's inside the box.

        Returns:
            [N, (y1, x1, y2, x2)] in pixel coordinates
        """
        h, w = shape
        scale = np.array([h - 1, w - 1, h - 1, w - 1])
        shift = np.array([0, 0, 1, 1])
        return np.around(np.multiply(boxes, scale) + shift).astype(np.int32)

    def resize(self, image, output_shape, order=1, mode='constant', cval=0, clip=True,
               preserve_range=False, anti_aliasing=False, anti_aliasing_sigma=None):
        """A wrapper for Scikit-Image resize().

        Scikit-Image generates warnings on every call to resize() if it doesn't
        receive the right parameters. The right parameters depend on the version
        of skimage. This solves the problem by using different parameters per
        version. And it provides a central place to control resizing defaults.
        """
        if LooseVersion(skimage.__version__) >= LooseVersion("0.14"):
            # New in 0.14: anti_aliasing. Default it to False for backward
            # compatibility with skimage 0.13.
            return skimage.transform.resize(
                image, output_shape,
                order=order, mode=mode, cval=cval, clip=clip,
                preserve_range=preserve_range, anti_aliasing=anti_aliasing,
                anti_aliasing_sigma=anti_aliasing_sigma)
        else:
            return skimage.transform.resize(
                image, output_shape,
                order=order, mode=mode, cval=cval, clip=clip,
                preserve_range=preserve_range)

    def mold_image(self, images, config):
        """Expects an RGB image (or array of images) and subtracts
        the mean pixel and converts it to float. Expects image
        colors in RGB order.
        """
        return images.astype(np.float32) - config.MEAN_PIXEL

    def compose_image_meta(self, image_id, original_image_shape, image_shape,
                           window, scale, active_class_ids):
        """Takes attributes of an image and puts them in one 1D array.

        image_id: An int ID of the image. Useful for debugging.
        original_image_shape: [H, W, C] before resizing or padding.
        image_shape: [H, W, C] after resizing and padding
        window: (y1, x1, y2, x2) in pixels. The area of the image where the real
                image is (excluding the padding)
        scale: The scaling factor applied to the original image (float32)
        active_class_ids: List of class_ids available in the dataset from which
            the image came. Useful if training on images from multiple datasets
            where not all classes are present in all datasets.
        """
        meta = np.array(
            [image_id] +  # size=1
            list(original_image_shape) +  # size=3
            list(image_shape) +  # size=3
            list(window) +  # size=4 (y1, x1, y2, x2) in image cooredinates
            [scale] +  # size=1
            list(active_class_ids)  # size=num_classes
        )
        return meta

    def compute_backbone_shapes(self, config, image_shape):
        """Computes the width and height of each stage of the backbone network.

        Returns:
            [N, (height, width)]. Where N is the number of stages
        """
        if callable(config.BACKBONE):
            return config.COMPUTE_BACKBONE_SHAPE(image_shape)

        # Currently supports ResNet only
        assert config.BACKBONE in ["resnet50", "resnet101"]
        return np.array(
            [[int(math.ceil(image_shape[0] / stride)),
              int(math.ceil(image_shape[1] / stride))]
             for stride in config.BACKBONE_STRIDES])


if __name__ == '__main__':
    image = skimage.io.imread('../images/25691390_f9944f61b5_z.jpg')

    m = MaskrcnnModel()
    final_rois, final_masks, final_class_ids, final_scores = m.inference(image)

    # Visualize results
    visualize.display_instances(image, final_rois, final_masks, final_class_ids,
                                m.class_names, final_scores)
