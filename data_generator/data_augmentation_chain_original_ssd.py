'''
The data augmentation operations of the original SSD implementation.

Copyright (C) 2018 Pierluigi Ferrari

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from __future__ import division
import numpy as np
import cv2
import inspect


from data_generator.object_detection_2d_image_boxes_validation_utils import BoundGenerator, BoxFilter, ImageValidator


import numpy as np
import cv2



class Resize:
    '''
    Resizes images to a specified height and width in pixels.
    '''

    def __init__(self,
                 height,
                 width,
                 interpolation_mode=cv2.INTER_LINEAR,
                 box_filter=None,
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        '''
        Arguments:
            height (int): The desired height of the output images in pixels.
            width (int): The desired width of the output images in pixels.
            interpolation_mode (int, optional): An integer that denotes a valid
                OpenCV interpolation mode. For example, integers 0 through 5 are
                valid interpolation modes.
            box_filter (BoxFilter, optional): Only relevant if ground truth bounding boxes are given.
                A `BoxFilter` object to filter out bounding boxes that don't meet the given criteria
                after the transformation. Refer to the `BoxFilter` documentation for details. If `None`,
                the validity of the bounding boxes is not checked.
            labels_format (dict, optional): A dictionary that defines which index in the last axis of the labels
                of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
        '''
        if not (isinstance(box_filter, BoxFilter) or box_filter is None):
            raise ValueError("`box_filter` must be either `None` or a `BoxFilter` object.")
        self.out_height = height
        self.out_width = width
        self.interpolation_mode = interpolation_mode
        self.box_filter = box_filter
        self.labels_format = labels_format

    def __call__(self, image, labels=None, return_inverter=False):

        img_height, img_width = image.shape[:2]

        xmin = self.labels_format['xmin']
        ymin = self.labels_format['ymin']
        xmax = self.labels_format['xmax']
        ymax = self.labels_format['ymax']

        image = cv2.resize(image,
                           dsize=(self.out_width, self.out_height),
                           interpolation=self.interpolation_mode)

        if return_inverter:
            def inverter(labels):
                labels = np.copy(labels)
                labels[:, [ymin+1, ymax+1]] = np.round(labels[:, [ymin+1, ymax+1]] * (img_height / self.out_height), decimals=0)
                labels[:, [xmin+1, xmax+1]] = np.round(labels[:, [xmin+1, xmax+1]] * (img_width / self.out_width), decimals=0)
                return labels

        if labels is None:
            if return_inverter:
                return image, inverter
            else:
                return image
        else:
            labels = np.copy(labels)
            labels[:, [ymin, ymax]] = np.round(labels[:, [ymin, ymax]] * (self.out_height / img_height), decimals=0)
            labels[:, [xmin, xmax]] = np.round(labels[:, [xmin, xmax]] * (self.out_width / img_width), decimals=0)

            if not (self.box_filter is None):
                self.box_filter.labels_format = self.labels_format
                labels = self.box_filter(labels=labels,
                                         image_height=self.out_height,
                                         image_width=self.out_width)

            if return_inverter:
                return image, labels, inverter
            else:
                return image, labels

class ResizeRandomInterp:
    '''
    Resizes images to a specified height and width in pixels using a radnomly
    selected interpolation mode.
    '''

    def __init__(self,
                 height,
                 width,
                 interpolation_modes=[cv2.INTER_NEAREST,
                                      cv2.INTER_LINEAR,
                                      cv2.INTER_CUBIC,
                                      cv2.INTER_AREA,
                                      cv2.INTER_LANCZOS4],
                 box_filter=None,
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        '''
        Arguments:
            height (int): The desired height of the output image in pixels.
            width (int): The desired width of the output image in pixels.
            interpolation_modes (list/tuple, optional): A list/tuple of integers
                that represent valid OpenCV interpolation modes. For example,
                integers 0 through 5 are valid interpolation modes.
            box_filter (BoxFilter, optional): Only relevant if ground truth bounding boxes are given.
                A `BoxFilter` object to filter out bounding boxes that don't meet the given criteria
                after the transformation. Refer to the `BoxFilter` documentation for details. If `None`,
                the validity of the bounding boxes is not checked.
            labels_format (dict, optional): A dictionary that defines which index in the last axis of the labels
                of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
        '''
        if not (isinstance(interpolation_modes, (list, tuple))):
            raise ValueError("`interpolation_mode` must be a list or tuple.")
        self.height = height
        self.width = width
        self.interpolation_modes = interpolation_modes
        self.box_filter = box_filter
        self.labels_format = labels_format
        self.resize = Resize(height=self.height,
                             width=self.width,
                             box_filter=self.box_filter,
                             labels_format=self.labels_format)

    def __call__(self, image, labels=None, return_inverter=False):
        self.resize.interpolation_mode = np.random.choice(self.interpolation_modes)
        self.resize.labels_format = self.labels_format
        return self.resize(image, labels, return_inverter)

class Flip:
    '''
    Flips images horizontally or vertically.
    '''
    def __init__(self,
                 dim='horizontal',
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        '''
        Arguments:
            dim (str, optional): Can be either of 'horizontal' and 'vertical'.
                If 'horizontal', images will be flipped horizontally, i.e. along
                the vertical axis. If 'horizontal', images will be flipped vertically,
                i.e. along the horizontal axis.
            labels_format (dict, optional): A dictionary that defines which index in the last axis of the labels
                of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
        '''
        if not (dim in {'horizontal', 'vertical'}): raise ValueError("`dim` can be one of 'horizontal' and 'vertical'.")
        self.dim = dim
        self.labels_format = labels_format

    def __call__(self, image, labels=None, return_inverter=False):

        img_height, img_width = image.shape[:2]

        xmin = self.labels_format['xmin']
        ymin = self.labels_format['ymin']
        xmax = self.labels_format['xmax']
        ymax = self.labels_format['ymax']

        if self.dim == 'horizontal':
            image = image[:,::-1]
            if labels is None:
                return image
            else:
                labels = np.copy(labels)
                labels[:, [xmin, xmax]] = img_width - labels[:, [xmax, xmin]]
                return image, labels
        else:
            image = image[::-1]
            if labels is None:
                return image
            else:
                labels = np.copy(labels)
                labels[:, [ymin, ymax]] = img_height - labels[:, [ymax, ymin]]
                return image, labels

class RandomFlip:
    '''
    Randomly flips images horizontally or vertically. The randomness only refers
    to whether or not the image will be flipped.
    '''
    def __init__(self,
                 dim='horizontal',
                 prob=0.5,
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        '''
        Arguments:
            dim (str, optional): Can be either of 'horizontal' and 'vertical'.
                If 'horizontal', images will be flipped horizontally, i.e. along
                the vertical axis. If 'horizontal', images will be flipped vertically,
                i.e. along the horizontal axis.
            prob (float, optional): `(1 - prob)` determines the probability with which the original,
                unaltered image is returned.
            labels_format (dict, optional): A dictionary that defines which index in the last axis of the labels
                of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
        '''
        self.dim = dim
        self.prob = prob
        self.labels_format = labels_format
        self.flip = Flip(dim=self.dim, labels_format=self.labels_format)

    def __call__(self, image, labels=None):
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            self.flip.labels_format = self.labels_format
            return self.flip(image, labels)
        elif labels is None:
            return image
        else:
            return image, labels

class Translate:
    '''
    Translates images horizontally and/or vertically.
    '''

    def __init__(self,
                 dy,
                 dx,
                 clip_boxes=True,
                 box_filter=None,
                 background=(0,0,0),
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        '''
        Arguments:
            dy (float): The fraction of the image height by which to translate images along the
                vertical axis. Positive values translate images downwards, negative values
                translate images upwards.
            dx (float): The fraction of the image width by which to translate images along the
                horizontal axis. Positive values translate images to the right, negative values
                translate images to the left.
            clip_boxes (bool, optional): Only relevant if ground truth bounding boxes are given.
                If `True`, any ground truth bounding boxes will be clipped to lie entirely within the
                image after the translation.
            box_filter (BoxFilter, optional): Only relevant if ground truth bounding boxes are given.
                A `BoxFilter` object to filter out bounding boxes that don't meet the given criteria
                after the transformation. Refer to the `BoxFilter` documentation for details. If `None`,
                the validity of the bounding boxes is not checked.
            background (list/tuple, optional): A 3-tuple specifying the RGB color value of the
                background pixels of the translated images.
            labels_format (dict, optional): A dictionary that defines which index in the last axis of the labels
                of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
        '''

        if not (isinstance(box_filter, BoxFilter) or box_filter is None):
            raise ValueError("`box_filter` must be either `None` or a `BoxFilter` object.")
        self.dy_rel = dy
        self.dx_rel = dx
        self.clip_boxes = clip_boxes
        self.box_filter = box_filter
        self.background = background
        self.labels_format = labels_format

    def __call__(self, image, labels=None):

        img_height, img_width = image.shape[:2]

        # Compute the translation matrix.
        dy_abs = int(round(img_height * self.dy_rel))
        dx_abs = int(round(img_width * self.dx_rel))
        M = np.float32([[1, 0, dx_abs],
                        [0, 1, dy_abs]])

        # Translate the image.
        image = cv2.warpAffine(image,
                               M=M,
                               dsize=(img_width, img_height),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=self.background)

        if labels is None:
            return image
        else:
            xmin = self.labels_format['xmin']
            ymin = self.labels_format['ymin']
            xmax = self.labels_format['xmax']
            ymax = self.labels_format['ymax']

            labels = np.copy(labels)
            # Translate the box coordinates to the translated image's coordinate system.
            labels[:,[xmin,xmax]] += dx_abs
            labels[:,[ymin,ymax]] += dy_abs

            # Compute all valid boxes for this patch.
            if not (self.box_filter is None):
                self.box_filter.labels_format = self.labels_format
                labels = self.box_filter(labels=labels,
                                         image_height=img_height,
                                         image_width=img_width)

            if self.clip_boxes:
                labels[:,[ymin,ymax]] = np.clip(labels[:,[ymin,ymax]], a_min=0, a_max=img_height-1)
                labels[:,[xmin,xmax]] = np.clip(labels[:,[xmin,xmax]], a_min=0, a_max=img_width-1)

            return image, labels

class RandomTranslate:
    '''
    Randomly translates images horizontally and/or vertically.
    '''

    def __init__(self,
                 dy_minmax=(0.03,0.3),
                 dx_minmax=(0.03,0.3),
                 prob=0.5,
                 clip_boxes=True,
                 box_filter=None,
                 image_validator=None,
                 n_trials_max=3,
                 background=(0,0,0),
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        '''
        Arguments:
            dy_minmax (list/tuple, optional): A 2-tuple `(min, max)` of non-negative floats that
                determines the minimum and maximum relative translation of images along the vertical
                axis both upward and downward. That is, images will be randomly translated by at least
                `min` and at most `max` either upward or downward. For example, if `dy_minmax == (0.05,0.3)`,
                an image of size `(100,100)` will be translated by at least 5 and at most 30 pixels
                either upward or downward. The translation direction is chosen randomly.
            dx_minmax (list/tuple, optional): A 2-tuple `(min, max)` of non-negative floats that
                determines the minimum and maximum relative translation of images along the horizontal
                axis both to the left and right. That is, images will be randomly translated by at least
                `min` and at most `max` either left or right. For example, if `dx_minmax == (0.05,0.3)`,
                an image of size `(100,100)` will be translated by at least 5 and at most 30 pixels
                either left or right. The translation direction is chosen randomly.
            prob (float, optional): `(1 - prob)` determines the probability with which the original,
                unaltered image is returned.
            clip_boxes (bool, optional): Only relevant if ground truth bounding boxes are given.
                If `True`, any ground truth bounding boxes will be clipped to lie entirely within the
                image after the translation.
            box_filter (BoxFilter, optional): Only relevant if ground truth bounding boxes are given.
                A `BoxFilter` object to filter out bounding boxes that don't meet the given criteria
                after the transformation. Refer to the `BoxFilter` documentation for details. If `None`,
                the validity of the bounding boxes is not checked.
            image_validator (ImageValidator, optional): Only relevant if ground truth bounding boxes are given.
                An `ImageValidator` object to determine whether a translated image is valid. If `None`,
                any outcome is valid.
            n_trials_max (int, optional): Only relevant if ground truth bounding boxes are given.
                Determines the maxmial number of trials to produce a valid image. If no valid image could
                be produced in `n_trials_max` trials, returns the unaltered input image.
            background (list/tuple, optional): A 3-tuple specifying the RGB color value of the
                background pixels of the translated images.
            labels_format (dict, optional): A dictionary that defines which index in the last axis of the labels
                of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
        '''
        if dy_minmax[0] > dy_minmax[1]:
            raise ValueError("It must be `dy_minmax[0] <= dy_minmax[1]`.")
        if dx_minmax[0] > dx_minmax[1]:
            raise ValueError("It must be `dx_minmax[0] <= dx_minmax[1]`.")
        if dy_minmax[0] < 0 or dx_minmax[0] < 0:
            raise ValueError("It must be `dy_minmax[0] >= 0` and `dx_minmax[0] >= 0`.")
        if not (isinstance(image_validator, ImageValidator) or image_validator is None):
            raise ValueError("`image_validator` must be either `None` or an `ImageValidator` object.")
        self.dy_minmax = dy_minmax
        self.dx_minmax = dx_minmax
        self.prob = prob
        self.clip_boxes = clip_boxes
        self.box_filter = box_filter
        self.image_validator = image_validator
        self.n_trials_max = n_trials_max
        self.background = background
        self.labels_format = labels_format
        self.translate = Translate(dy=0,
                                   dx=0,
                                   clip_boxes=self.clip_boxes,
                                   box_filter=self.box_filter,
                                   background=self.background,
                                   labels_format=self.labels_format)

    def __call__(self, image, labels=None):

        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):

            img_height, img_width = image.shape[:2]

            xmin = self.labels_format['xmin']
            ymin = self.labels_format['ymin']
            xmax = self.labels_format['xmax']
            ymax = self.labels_format['ymax']

            # Override the preset labels format.
            if not self.image_validator is None:
                self.image_validator.labels_format = self.labels_format
            self.translate.labels_format = self.labels_format

            for _ in range(max(1, self.n_trials_max)):

                # Pick the relative amount by which to translate.
                dy_abs = np.random.uniform(self.dy_minmax[0], self.dy_minmax[1])
                dx_abs = np.random.uniform(self.dx_minmax[0], self.dx_minmax[1])
                # Pick the direction in which to translate.
                dy = np.random.choice([-dy_abs, dy_abs])
                dx = np.random.choice([-dx_abs, dx_abs])
                self.translate.dy_rel = dy
                self.translate.dx_rel = dx

                if (labels is None) or (self.image_validator is None):
                    # We either don't have any boxes or if we do, we will accept any outcome as valid.
                    return self.translate(image, labels)
                else:
                    # Translate the box coordinates to the translated image's coordinate system.
                    new_labels = np.copy(labels)
                    new_labels[:, [ymin, ymax]] += int(round(img_height * dy))
                    new_labels[:, [xmin, xmax]] += int(round(img_width * dx))

                    # Check if the patch is valid.
                    if self.image_validator(labels=new_labels,
                                            image_height=img_height,
                                            image_width=img_width):
                        return self.translate(image, labels)

            # If all attempts failed, return the unaltered input image.
            if labels is None:
                return image

            else:
                return image, labels

        elif labels is None:
            return image

        else:
            return image, labels

class Scale:
    '''
    Scales images, i.e. zooms in or out.
    '''

    def __init__(self,
                 factor,
                 clip_boxes=True,
                 box_filter=None,
                 background=(0,0,0),
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        '''
        Arguments:
            factor (float): The fraction of the image size by which to scale images. Must be positive.
            clip_boxes (bool, optional): Only relevant if ground truth bounding boxes are given.
                If `True`, any ground truth bounding boxes will be clipped to lie entirely within the
                image after the translation.
            box_filter (BoxFilter, optional): Only relevant if ground truth bounding boxes are given.
                A `BoxFilter` object to filter out bounding boxes that don't meet the given criteria
                after the transformation. Refer to the `BoxFilter` documentation for details. If `None`,
                the validity of the bounding boxes is not checked.
            background (list/tuple, optional): A 3-tuple specifying the RGB color value of the potential
                background pixels of the scaled images.
            labels_format (dict, optional): A dictionary that defines which index in the last axis of the labels
                of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
        '''

        if factor <= 0:
            raise ValueError("It must be `factor > 0`.")
        if not (isinstance(box_filter, BoxFilter) or box_filter is None):
            raise ValueError("`box_filter` must be either `None` or a `BoxFilter` object.")
        self.factor = factor
        self.clip_boxes = clip_boxes
        self.box_filter = box_filter
        self.background = background
        self.labels_format = labels_format

    def __call__(self, image, labels=None):

        img_height, img_width = image.shape[:2]

        # Compute the rotation matrix.
        M = cv2.getRotationMatrix2D(center=(img_width / 2, img_height / 2),
                                    angle=0,
                                    scale=self.factor)

        # Scale the image.
        image = cv2.warpAffine(image,
                               M=M,
                               dsize=(img_width, img_height),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=self.background)

        if labels is None:
            return image
        else:
            xmin = self.labels_format['xmin']
            ymin = self.labels_format['ymin']
            xmax = self.labels_format['xmax']
            ymax = self.labels_format['ymax']

            labels = np.copy(labels)
            # Scale the bounding boxes accordingly.
            # Transform two opposite corner points of the rectangular boxes using the rotation matrix `M`.
            toplefts = np.array([labels[:,xmin], labels[:,ymin], np.ones(labels.shape[0])])
            bottomrights = np.array([labels[:,xmax], labels[:,ymax], np.ones(labels.shape[0])])
            new_toplefts = (np.dot(M, toplefts)).T
            new_bottomrights = (np.dot(M, bottomrights)).T
            labels[:,[xmin,ymin]] = np.round(new_toplefts, decimals=0).astype(np.int)
            labels[:,[xmax,ymax]] = np.round(new_bottomrights, decimals=0).astype(np.int)

            # Compute all valid boxes for this patch.
            if not (self.box_filter is None):
                self.box_filter.labels_format = self.labels_format
                labels = self.box_filter(labels=labels,
                                         image_height=img_height,
                                         image_width=img_width)

            if self.clip_boxes:
                labels[:,[ymin,ymax]] = np.clip(labels[:,[ymin,ymax]], a_min=0, a_max=img_height-1)
                labels[:,[xmin,xmax]] = np.clip(labels[:,[xmin,xmax]], a_min=0, a_max=img_width-1)

            return image, labels

class RandomScale:
    '''
    Randomly scales images.
    '''

    def __init__(self,
                 min_factor=0.5,
                 max_factor=1.5,
                 prob=0.5,
                 clip_boxes=True,
                 box_filter=None,
                 image_validator=None,
                 n_trials_max=3,
                 background=(0,0,0),
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        '''
        Arguments:
            min_factor (float, optional): The minimum fraction of the image size by which to scale images.
                Must be positive.
            max_factor (float, optional): The maximum fraction of the image size by which to scale images.
                Must be positive.
            prob (float, optional): `(1 - prob)` determines the probability with which the original,
                unaltered image is returned.
            clip_boxes (bool, optional): Only relevant if ground truth bounding boxes are given.
                If `True`, any ground truth bounding boxes will be clipped to lie entirely within the
                image after the translation.
            box_filter (BoxFilter, optional): Only relevant if ground truth bounding boxes are given.
                A `BoxFilter` object to filter out bounding boxes that don't meet the given criteria
                after the transformation. Refer to the `BoxFilter` documentation for details. If `None`,
                the validity of the bounding boxes is not checked.
            image_validator (ImageValidator, optional): Only relevant if ground truth bounding boxes are given.
                An `ImageValidator` object to determine whether a scaled image is valid. If `None`,
                any outcome is valid.
            n_trials_max (int, optional): Only relevant if ground truth bounding boxes are given.
                Determines the maxmial number of trials to produce a valid image. If no valid image could
                be produced in `n_trials_max` trials, returns the unaltered input image.
            background (list/tuple, optional): A 3-tuple specifying the RGB color value of the potential
                background pixels of the scaled images.
            labels_format (dict, optional): A dictionary that defines which index in the last axis of the labels
                of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
        '''

        if not (0 < min_factor <= max_factor):
            raise ValueError("It must be `0 < min_factor <= max_factor`.")
        if not (isinstance(image_validator, ImageValidator) or image_validator is None):
            raise ValueError("`image_validator` must be either `None` or an `ImageValidator` object.")
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.prob = prob
        self.clip_boxes = clip_boxes
        self.box_filter = box_filter
        self.image_validator = image_validator
        self.n_trials_max = n_trials_max
        self.background = background
        self.labels_format = labels_format
        self.scale = Scale(factor=1.0,
                           clip_boxes=self.clip_boxes,
                           box_filter=self.box_filter,
                           background=self.background,
                           labels_format=self.labels_format)

    def __call__(self, image, labels=None):

        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):

            img_height, img_width = image.shape[:2]

            xmin = self.labels_format['xmin']
            ymin = self.labels_format['ymin']
            xmax = self.labels_format['xmax']
            ymax = self.labels_format['ymax']

            # Override the preset labels format.
            if not self.image_validator is None:
                self.image_validator.labels_format = self.labels_format
            self.scale.labels_format = self.labels_format

            for _ in range(max(1, self.n_trials_max)):

                # Pick a scaling factor.
                factor = np.random.uniform(self.min_factor, self.max_factor)
                self.scale.factor = factor

                if (labels is None) or (self.image_validator is None):
                    # We either don't have any boxes or if we do, we will accept any outcome as valid.
                    return self.scale(image, labels)
                else:
                    # Scale the bounding boxes accordingly.
                    # Transform two opposite corner points of the rectangular boxes using the rotation matrix `M`.
                    toplefts = np.array([labels[:,xmin], labels[:,ymin], np.ones(labels.shape[0])])
                    bottomrights = np.array([labels[:,xmax], labels[:,ymax], np.ones(labels.shape[0])])

                    # Compute the rotation matrix.
                    M = cv2.getRotationMatrix2D(center=(img_width / 2, img_height / 2),
                                                angle=0,
                                                scale=factor)

                    new_toplefts = (np.dot(M, toplefts)).T
                    new_bottomrights = (np.dot(M, bottomrights)).T

                    new_labels = np.copy(labels)
                    new_labels[:,[xmin,ymin]] = np.around(new_toplefts, decimals=0).astype(np.int)
                    new_labels[:,[xmax,ymax]] = np.around(new_bottomrights, decimals=0).astype(np.int)

                    # Check if the patch is valid.
                    if self.image_validator(labels=new_labels,
                                            image_height=img_height,
                                            image_width=img_width):
                        return self.scale(image, labels)

            # If all attempts failed, return the unaltered input image.
            if labels is None:
                return image

            else:
                return image, labels

        elif labels is None:
            return image

        else:
            return image, labels

class Rotate:
    '''
    Rotates images counter-clockwise by 90, 180, or 270 degrees.
    '''

    def __init__(self,
                 angle,
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        '''
        Arguments:
            angle (int): The angle in degrees by which to rotate the images counter-clockwise.
                Only 90, 180, and 270 are valid values.
            labels_format (dict, optional): A dictionary that defines which index in the last axis of the labels
                of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
        '''

        if not angle in {90, 180, 270}:
            raise ValueError("`angle` must be in the set {90, 180, 270}.")
        self.angle = angle
        self.labels_format = labels_format

    def __call__(self, image, labels=None):

        img_height, img_width = image.shape[:2]

        # Compute the rotation matrix.
        M = cv2.getRotationMatrix2D(center=(img_width / 2, img_height / 2),
                                    angle=self.angle,
                                    scale=1)

        # Get the sine and cosine from the rotation matrix.
        cos_angle = np.abs(M[0, 0])
        sin_angle = np.abs(M[0, 1])

        # Compute the new bounding dimensions of the image.
        img_width_new = int(img_height * sin_angle + img_width * cos_angle)
        img_height_new = int(img_height * cos_angle + img_width * sin_angle)

        # Adjust the rotation matrix to take into account the translation.
        M[1, 2] += (img_height_new - img_height) / 2
        M[0, 2] += (img_width_new - img_width) / 2

        # Rotate the image.
        image = cv2.warpAffine(image,
                               M=M,
                               dsize=(img_width_new, img_height_new))

        if labels is None:
            return image
        else:
            xmin = self.labels_format['xmin']
            ymin = self.labels_format['ymin']
            xmax = self.labels_format['xmax']
            ymax = self.labels_format['ymax']

            labels = np.copy(labels)
            # Rotate the bounding boxes accordingly.
            # Transform two opposite corner points of the rectangular boxes using the rotation matrix `M`.
            toplefts = np.array([labels[:,xmin], labels[:,ymin], np.ones(labels.shape[0])])
            bottomrights = np.array([labels[:,xmax], labels[:,ymax], np.ones(labels.shape[0])])
            new_toplefts = (np.dot(M, toplefts)).T
            new_bottomrights = (np.dot(M, bottomrights)).T
            labels[:,[xmin,ymin]] = np.round(new_toplefts, decimals=0).astype(np.int)
            labels[:,[xmax,ymax]] = np.round(new_bottomrights, decimals=0).astype(np.int)

            if self.angle == 90:
                # ymin and ymax were switched by the rotation.
                labels[:,[ymax,ymin]] = labels[:,[ymin,ymax]]
            elif self.angle == 180:
                # ymin and ymax were switched by the rotation,
                # and also xmin and xmax were switched.
                labels[:,[ymax,ymin]] = labels[:,[ymin,ymax]]
                labels[:,[xmax,xmin]] = labels[:,[xmin,xmax]]
            elif self.angle == 270:
                # xmin and xmax were switched by the rotation.
                labels[:,[xmax,xmin]] = labels[:,[xmin,xmax]]

            return image, labels

class RandomRotate:
    '''
    Randomly rotates images counter-clockwise.
    '''

    def __init__(self,
                 angles=[90, 180, 270],
                 prob=0.5,
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        '''
        Arguments:
            angle (list): The list of angles in degrees from which one is randomly selected to rotate
                the images counter-clockwise. Only 90, 180, and 270 are valid values.
            prob (float, optional): `(1 - prob)` determines the probability with which the original,
                unaltered image is returned.
            labels_format (dict, optional): A dictionary that defines which index in the last axis of the labels
                of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
        '''
        for angle in angles:
            if not angle in {90, 180, 270}:
                raise ValueError("`angles` can only contain the values 90, 180, and 270.")
        self.angles = angles
        self.prob = prob
        self.labels_format = labels_format
        self.rotate = Rotate(angle=90, labels_format=self.labels_format)

    def __call__(self, image, labels=None):

        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            # Pick a rotation angle.
            self.rotate.angle = random.choice(self.angles)
            self.rotate.labels_format = self.labels_format
            return self.rotate(image, labels)

        elif labels is None:
            return image

        else:
            return image, labels


class PatchCoordinateGenerator:
    '''
    Generates random patch coordinates that meet specified requirements.
    '''

    def __init__(self,
                 img_height=None,
                 img_width=None,
                 must_match='h_w',
                 min_scale=0.3,
                 max_scale=1.0,
                 scale_uniformly=False,
                 min_aspect_ratio = 0.5,
                 max_aspect_ratio = 2.0,
                 patch_ymin=None,
                 patch_xmin=None,
                 patch_height=None,
                 patch_width=None,
                 patch_aspect_ratio=None):
        '''
        Arguments:
            img_height (int): The height of the image for which the patch coordinates
                shall be generated. Doesn't have to be known upon construction.
            img_width (int): The width of the image for which the patch coordinates
                shall be generated. Doesn't have to be known upon construction.
            must_match (str, optional): Can be either of 'h_w', 'h_ar', and 'w_ar'.
                Specifies which two of the three quantities height, width, and aspect
                ratio determine the shape of the generated patch. The respective third
                quantity will be computed from the other two. For example,
                if `must_match == 'h_w'`, then the patch's height and width will be
                set to lie within [min_scale, max_scale] of the image size or to
                `patch_height` and/or `patch_width`, if given. The patch's aspect ratio
                is the dependent variable in this case, it will be computed from the
                height and width. Any given values for `patch_aspect_ratio`,
                `min_aspect_ratio`, or `max_aspect_ratio` will be ignored.
            min_scale (float, optional): The minimum size of a dimension of the patch
                as a fraction of the respective dimension of the image. Can be greater
                than 1. For example, if the image width is 200 and `min_scale == 0.5`,
                then the width of the generated patch will be at least 100. If `min_scale == 1.5`,
                the width of the generated patch will be at least 300.
            max_scale (float, optional): The maximum size of a dimension of the patch
                as a fraction of the respective dimension of the image. Can be greater
                than 1. For example, if the image width is 200 and `max_scale == 1.0`,
                then the width of the generated patch will be at most 200. If `max_scale == 1.5`,
                the width of the generated patch will be at most 300. Must be greater than
                `min_scale`.
            scale_uniformly (bool, optional): If `True` and if `must_match == 'h_w'`,
                the patch height and width will be scaled uniformly, otherwise they will
                be scaled independently.
            min_aspect_ratio (float, optional): Determines the minimum aspect ratio
                for the generated patches.
            max_aspect_ratio (float, optional): Determines the maximum aspect ratio
                for the generated patches.
            patch_ymin (int, optional): `None` or the vertical coordinate of the top left
                corner of the generated patches. If this is not `None`, the position of the
                patches along the vertical axis is fixed. If this is `None`, then the
                vertical position of generated patches will be chosen randomly such that
                the overlap of a patch and the image along the vertical dimension is
                always maximal.
            patch_xmin (int, optional): `None` or the horizontal coordinate of the top left
                corner of the generated patches. If this is not `None`, the position of the
                patches along the horizontal axis is fixed. If this is `None`, then the
                horizontal position of generated patches will be chosen randomly such that
                the overlap of a patch and the image along the horizontal dimension is
                always maximal.
            patch_height (int, optional): `None` or the fixed height of the generated patches.
            patch_width (int, optional): `None` or the fixed width of the generated patches.
            patch_aspect_ratio (float, optional): `None` or the fixed aspect ratio of the
                generated patches.
        '''

        if not (must_match in {'h_w', 'h_ar', 'w_ar'}):
            raise ValueError("`must_match` must be either of 'h_w', 'h_ar' and 'w_ar'.")
        if min_scale >= max_scale:
            raise ValueError("It must be `min_scale < max_scale`.")
        if min_aspect_ratio >= max_aspect_ratio:
            raise ValueError("It must be `min_aspect_ratio < max_aspect_ratio`.")
        if scale_uniformly and not ((patch_height is None) and (patch_width is None)):
            raise ValueError("If `scale_uniformly == True`, `patch_height` and `patch_width` must both be `None`.")
        self.img_height = img_height
        self.img_width = img_width
        self.must_match = must_match
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scale_uniformly = scale_uniformly
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.patch_ymin = patch_ymin
        self.patch_xmin = patch_xmin
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.patch_aspect_ratio = patch_aspect_ratio

    def __call__(self):
        '''
        Returns:
            A 4-tuple `(ymin, xmin, height, width)` that represents the coordinates
            of the generated patch.
        '''

        # Get the patch height and width.

        if self.must_match == 'h_w': # Aspect is the dependent variable.
            if not self.scale_uniformly:
                # Get the height.
                if self.patch_height is None:
                    patch_height = int(np.random.uniform(self.min_scale, self.max_scale) * self.img_height)
                else:
                    patch_height = self.patch_height
                # Get the width.
                if self.patch_width is None:
                    patch_width = int(np.random.uniform(self.min_scale, self.max_scale) * self.img_width)
                else:
                    patch_width = self.patch_width
            else:
                scaling_factor = np.random.uniform(self.min_scale, self.max_scale)
                patch_height = int(scaling_factor * self.img_height)
                patch_width = int(scaling_factor * self.img_width)

        elif self.must_match == 'h_ar': # Width is the dependent variable.
            # Get the height.
            if self.patch_height is None:
                patch_height = int(np.random.uniform(self.min_scale, self.max_scale) * self.img_height)
            else:
                patch_height = self.patch_height
            # Get the aspect ratio.
            if self.patch_aspect_ratio is None:
                patch_aspect_ratio = np.random.uniform(self.min_aspect_ratio, self.max_aspect_ratio)
            else:
                patch_aspect_ratio = self.patch_aspect_ratio
            # Get the width.
            patch_width = int(patch_height * patch_aspect_ratio)

        elif self.must_match == 'w_ar': # Height is the dependent variable.
            # Get the width.
            if self.patch_width is None:
                patch_width = int(np.random.uniform(self.min_scale, self.max_scale) * self.img_width)
            else:
                patch_width = self.patch_width
            # Get the aspect ratio.
            if self.patch_aspect_ratio is None:
                patch_aspect_ratio = np.random.uniform(self.min_aspect_ratio, self.max_aspect_ratio)
            else:
                patch_aspect_ratio = self.patch_aspect_ratio
            # Get the height.
            patch_height = int(patch_width / patch_aspect_ratio)

        # Get the top left corner coordinates of the patch.

        if self.patch_ymin is None:
            # Compute how much room we have along the vertical axis to place the patch.
            # A negative number here means that we want to sample a patch that is larger than the original image
            # in the vertical dimension, in which case the patch will be placed such that it fully contains the
            # image in the vertical dimension.
            y_range = self.img_height - patch_height
            # Select a random top left corner for the sample position from the possible positions.
            if y_range >= 0: patch_ymin = np.random.randint(0, y_range + 1) # There are y_range + 1 possible positions for the crop in the vertical dimension.
            else: patch_ymin = np.random.randint(y_range, 1) # The possible positions for the image on the background canvas in the vertical dimension.
        else:
            patch_ymin = self.patch_ymin

        if self.patch_xmin is None:
            # Compute how much room we have along the horizontal axis to place the patch.
            # A negative number here means that we want to sample a patch that is larger than the original image
            # in the horizontal dimension, in which case the patch will be placed such that it fully contains the
            # image in the horizontal dimension.
            x_range = self.img_width - patch_width
            # Select a random top left corner for the sample position from the possible positions.
            if x_range >= 0: patch_xmin = np.random.randint(0, x_range + 1) # There are x_range + 1 possible positions for the crop in the horizontal dimension.
            else: patch_xmin = np.random.randint(x_range, 1) # The possible positions for the image on the background canvas in the horizontal dimension.
        else:
            patch_xmin = self.patch_xmin

        return (patch_ymin, patch_xmin, patch_height, patch_width)

class CropPad:
    '''
    Crops and/or pads an image deterministically.

    Depending on the given output patch size and the position (top left corner) relative
    to the input image, the image will be cropped and/or padded along one or both spatial
    dimensions.

    For example, if the output patch lies entirely within the input image, this will result
    in a regular crop. If the input image lies entirely within the output patch, this will
    result in the image being padded in every direction. All other cases are mixed cases
    where the image might be cropped in some directions and padded in others.

    The output patch can be arbitrary in both size and position as long as it overlaps
    with the input image.
    '''

    def __init__(self,
                 patch_ymin,
                 patch_xmin,
                 patch_height,
                 patch_width,
                 clip_boxes=True,
                 box_filter=None,
                 background=(0,0,0),
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        '''
        Arguments:
            patch_ymin (int, optional): The vertical coordinate of the top left corner of the output
                patch relative to the image coordinate system. Can be negative (i.e. lie outside the image)
                as long as the resulting patch still overlaps with the image.
            patch_ymin (int, optional): The horizontal coordinate of the top left corner of the output
                patch relative to the image coordinate system. Can be negative (i.e. lie outside the image)
                as long as the resulting patch still overlaps with the image.
            patch_height (int): The height of the patch to be sampled from the image. Can be greater
                than the height of the input image.
            patch_width (int): The width of the patch to be sampled from the image. Can be greater
                than the width of the input image.
            clip_boxes (bool, optional): Only relevant if ground truth bounding boxes are given.
                If `True`, any ground truth bounding boxes will be clipped to lie entirely within the
                sampled patch.
            box_filter (BoxFilter, optional): Only relevant if ground truth bounding boxes are given.
                A `BoxFilter` object to filter out bounding boxes that don't meet the given criteria
                after the transformation. Refer to the `BoxFilter` documentation for details. If `None`,
                the validity of the bounding boxes is not checked.
            background (list/tuple, optional): A 3-tuple specifying the RGB color value of the potential
                background pixels of the scaled images. In the case of single-channel images,
                the first element of `background` will be used as the background pixel value.
            labels_format (dict, optional): A dictionary that defines which index in the last axis of the labels
                of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
        '''
        #if (patch_height <= 0) or (patch_width <= 0):
        #    raise ValueError("Patch height and width must both be positive.")
        #if (patch_ymin + patch_height < 0) or (patch_xmin + patch_width < 0):
        #    raise ValueError("A patch with the given coordinates cannot overlap with an input image.")
        if not (isinstance(box_filter, BoxFilter) or box_filter is None):
            raise ValueError("`box_filter` must be either `None` or a `BoxFilter` object.")
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.patch_ymin = patch_ymin
        self.patch_xmin = patch_xmin
        self.clip_boxes = clip_boxes
        self.box_filter = box_filter
        self.background = background
        self.labels_format = labels_format

    def __call__(self, image, labels=None, return_inverter=False):

        img_height, img_width = image.shape[:2]

        if (self.patch_ymin > img_height) or (self.patch_xmin > img_width):
            raise ValueError("The given patch doesn't overlap with the input image.")

        labels = np.copy(labels)

        xmin = self.labels_format['xmin']
        ymin = self.labels_format['ymin']
        xmax = self.labels_format['xmax']
        ymax = self.labels_format['ymax']

        # Top left corner of the patch relative to the image coordinate system:
        patch_ymin = self.patch_ymin
        patch_xmin = self.patch_xmin

        # Create a canvas of the size of the patch we want to end up with.
        if image.ndim == 3:
            canvas = np.zeros(shape=(self.patch_height, self.patch_width, 3), dtype=np.uint8)
            canvas[:, :] = self.background
        elif image.ndim == 2:
            canvas = np.zeros(shape=(self.patch_height, self.patch_width), dtype=np.uint8)
            canvas[:, :] = self.background[0]

        # Perform the crop.
        if patch_ymin < 0 and patch_xmin < 0: # Pad the image at the top and on the left.
            image_crop_height = min(img_height, self.patch_height + patch_ymin)  # The number of pixels of the image that will end up on the canvas in the vertical direction.
            image_crop_width = min(img_width, self.patch_width + patch_xmin) # The number of pixels of the image that will end up on the canvas in the horizontal direction.
            canvas[-patch_ymin:-patch_ymin + image_crop_height, -patch_xmin:-patch_xmin + image_crop_width] = image[:image_crop_height, :image_crop_width]

        elif patch_ymin < 0 and patch_xmin >= 0: # Pad the image at the top and crop it on the left.
            image_crop_height = min(img_height, self.patch_height + patch_ymin)  # The number of pixels of the image that will end up on the canvas in the vertical direction.
            image_crop_width = min(self.patch_width, img_width - patch_xmin) # The number of pixels of the image that will end up on the canvas in the horizontal direction.
            canvas[-patch_ymin:-patch_ymin + image_crop_height, :image_crop_width] = image[:image_crop_height, patch_xmin:patch_xmin + image_crop_width]

        elif patch_ymin >= 0 and patch_xmin < 0: # Crop the image at the top and pad it on the left.
            image_crop_height = min(self.patch_height, img_height - patch_ymin) # The number of pixels of the image that will end up on the canvas in the vertical direction.
            image_crop_width = min(img_width, self.patch_width + patch_xmin) # The number of pixels of the image that will end up on the canvas in the horizontal direction.
            canvas[:image_crop_height, -patch_xmin:-patch_xmin + image_crop_width] = image[patch_ymin:patch_ymin + image_crop_height, :image_crop_width]

        elif patch_ymin >= 0 and patch_xmin >= 0: # Crop the image at the top and on the left.
            image_crop_height = min(self.patch_height, img_height - patch_ymin) # The number of pixels of the image that will end up on the canvas in the vertical direction.
            image_crop_width = min(self.patch_width, img_width - patch_xmin) # The number of pixels of the image that will end up on the canvas in the horizontal direction.
            canvas[:image_crop_height, :image_crop_width] = image[patch_ymin:patch_ymin + image_crop_height, patch_xmin:patch_xmin + image_crop_width]

        image = canvas

        if return_inverter:
            def inverter(labels):
                labels = np.copy(labels)
                labels[:, [ymin+1, ymax+1]] += patch_ymin
                labels[:, [xmin+1, xmax+1]] += patch_xmin
                return labels

        if not (labels is None):

            # Translate the box coordinates to the patch's coordinate system.
            labels[:, [ymin, ymax]] -= patch_ymin
            labels[:, [xmin, xmax]] -= patch_xmin

            # Compute all valid boxes for this patch.
            if not (self.box_filter is None):
                self.box_filter.labels_format = self.labels_format
                labels = self.box_filter(labels=labels,
                                         image_height=self.patch_height,
                                         image_width=self.patch_width)

            if self.clip_boxes:
                labels[:,[ymin,ymax]] = np.clip(labels[:,[ymin,ymax]], a_min=0, a_max=self.patch_height-1)
                labels[:,[xmin,xmax]] = np.clip(labels[:,[xmin,xmax]], a_min=0, a_max=self.patch_width-1)

            if return_inverter:
                return image, labels, inverter
            else:
                return image, labels

        else:
            if return_inverter:
                return image, inverter
            else:
                return image

class Crop:
    '''
    Crops off the specified numbers of pixels from the borders of images.

    This is just a convenience interface for `CropPad`.
    '''

    def __init__(self,
                 crop_top,
                 crop_bottom,
                 crop_left,
                 crop_right,
                 clip_boxes=True,
                 box_filter=None,
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        self.crop_top = crop_top
        self.crop_bottom = crop_bottom
        self.crop_left = crop_left
        self.crop_right = crop_right
        self.clip_boxes = clip_boxes
        self.box_filter = box_filter
        self.labels_format = labels_format
        self.crop = CropPad(patch_ymin=self.crop_top,
                            patch_xmin=self.crop_left,
                            patch_height=None,
                            patch_width=None,
                            clip_boxes=self.clip_boxes,
                            box_filter=self.box_filter,
                            labels_format=self.labels_format)

    def __call__(self, image, labels=None, return_inverter=False):

        img_height, img_width = image.shape[:2]

        self.crop.patch_height = img_height - self.crop_top - self.crop_bottom
        self.crop.patch_width = img_width - self.crop_left - self.crop_right
        self.crop.labels_format = self.labels_format

        return self.crop(image, labels, return_inverter)

class Pad:
    '''
    Pads images by the specified numbers of pixels on each side.

    This is just a convenience interface for `CropPad`.
    '''

    def __init__(self,
                 pad_top,
                 pad_bottom,
                 pad_left,
                 pad_right,
                 background=(0,0,0),
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        self.pad_top = pad_top
        self.pad_bottom = pad_bottom
        self.pad_left = pad_left
        self.pad_right = pad_right
        self.background = background
        self.labels_format = labels_format
        self.pad = CropPad(patch_ymin=-self.pad_top,
                           patch_xmin=-self.pad_left,
                           patch_height=None,
                           patch_width=None,
                           clip_boxes=False,
                           box_filter=None,
                           background=self.background,
                           labels_format=self.labels_format)

    def __call__(self, image, labels=None, return_inverter=False):

        img_height, img_width = image.shape[:2]

        self.pad.patch_height = img_height + self.pad_top + self.pad_bottom
        self.pad.patch_width = img_width + self.pad_left + self.pad_right
        self.pad.labels_format = self.labels_format

        return self.pad(image, labels, return_inverter)

class RandomPatch:
    '''
    Randomly samples a patch from an image. The randomness refers to whatever
    randomness may be introduced by the patch coordinate generator, the box filter,
    and the patch validator.

    Input images may be cropped and/or padded along either or both of the two
    spatial dimensions as necessary in order to obtain the required patch.

    As opposed to `RandomPatchInf`, it is possible for this transform to fail to produce
    an output image at all, in which case it will return `None`. This is useful, because
    if this transform is used to generate patches of a fixed size or aspect ratio, then
    the caller needs to be able to rely on the output image satisfying the set size or
    aspect ratio. It might therefore not be an option to return the unaltered input image
    as other random transforms do when they fail to produce a valid transformed image.
    '''

    def __init__(self,
                 patch_coord_generator,
                 box_filter=None,
                 image_validator=None,
                 n_trials_max=3,
                 clip_boxes=True,
                 prob=1.0,
                 background=(0,0,0),
                 can_fail=False,
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        '''
        Arguments:
            patch_coord_generator (PatchCoordinateGenerator): A `PatchCoordinateGenerator` object
                to generate the positions and sizes of the patches to be sampled from the input images.
            box_filter (BoxFilter, optional): Only relevant if ground truth bounding boxes are given.
                A `BoxFilter` object to filter out bounding boxes that don't meet the given criteria
                after the transformation. Refer to the `BoxFilter` documentation for details. If `None`,
                the validity of the bounding boxes is not checked.
            image_validator (ImageValidator, optional): Only relevant if ground truth bounding boxes are given.
                An `ImageValidator` object to determine whether a sampled patch is valid. If `None`,
                any outcome is valid.
            n_trials_max (int, optional): Only relevant if ground truth bounding boxes are given.
                Determines the maxmial number of trials to sample a valid patch. If no valid patch could
                be sampled in `n_trials_max` trials, returns one `None` in place of each regular output.
            clip_boxes (bool, optional): Only relevant if ground truth bounding boxes are given.
                If `True`, any ground truth bounding boxes will be clipped to lie entirely within the
                sampled patch.
            prob (float, optional): `(1 - prob)` determines the probability with which the original,
                unaltered image is returned.
            background (list/tuple, optional): A 3-tuple specifying the RGB color value of the potential
                background pixels of the scaled images. In the case of single-channel images,
                the first element of `background` will be used as the background pixel value.
            can_fail (bool, optional): If `True`, will return `None` if no valid patch could be found after
                `n_trials_max` trials. If `False`, will return the unaltered input image in such a case.
            labels_format (dict, optional): A dictionary that defines which index in the last axis of the labels
                of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
        '''
        if not isinstance(patch_coord_generator, PatchCoordinateGenerator):
            raise ValueError("`patch_coord_generator` must be an instance of `PatchCoordinateGenerator`.")
        if not (isinstance(image_validator, ImageValidator) or image_validator is None):
            raise ValueError("`image_validator` must be either `None` or an `ImageValidator` object.")
        self.patch_coord_generator = patch_coord_generator
        self.box_filter = box_filter
        self.image_validator = image_validator
        self.n_trials_max = n_trials_max
        self.clip_boxes = clip_boxes
        self.prob = prob
        self.background = background
        self.can_fail = can_fail
        self.labels_format = labels_format
        self.sample_patch = CropPad(patch_ymin=None,
                                    patch_xmin=None,
                                    patch_height=None,
                                    patch_width=None,
                                    clip_boxes=self.clip_boxes,
                                    box_filter=self.box_filter,
                                    background=self.background,
                                    labels_format=self.labels_format)

    def __call__(self, image, labels=None, return_inverter=False):

        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):

            img_height, img_width = image.shape[:2]
            self.patch_coord_generator.img_height = img_height
            self.patch_coord_generator.img_width = img_width

            xmin = self.labels_format['xmin']
            ymin = self.labels_format['ymin']
            xmax = self.labels_format['xmax']
            ymax = self.labels_format['ymax']

            # Override the preset labels format.
            if not self.image_validator is None:
                self.image_validator.labels_format = self.labels_format
            self.sample_patch.labels_format = self.labels_format

            for _ in range(max(1, self.n_trials_max)):

                # Generate patch coordinates.
                patch_ymin, patch_xmin, patch_height, patch_width = self.patch_coord_generator()

                self.sample_patch.patch_ymin = patch_ymin
                self.sample_patch.patch_xmin = patch_xmin
                self.sample_patch.patch_height = patch_height
                self.sample_patch.patch_width = patch_width

                if (labels is None) or (self.image_validator is None):
                    # We either don't have any boxes or if we do, we will accept any outcome as valid.
                    return self.sample_patch(image, labels, return_inverter)
                else:
                    # Translate the box coordinates to the patch's coordinate system.
                    new_labels = np.copy(labels)
                    new_labels[:, [ymin, ymax]] -= patch_ymin
                    new_labels[:, [xmin, xmax]] -= patch_xmin
                    # Check if the patch is valid.
                    if self.image_validator(labels=new_labels,
                                            image_height=patch_height,
                                            image_width=patch_width):
                        return self.sample_patch(image, labels, return_inverter)

            # If we weren't able to sample a valid patch...
            if self.can_fail:
                # ...return `None`.
                if labels is None:
                    if return_inverter:
                        return None, None
                    else:
                        return None
                else:
                    if return_inverter:
                        return None, None, None
                    else:
                        return None, None
            else:
                # ...return the unaltered input image.
                if labels is None:
                    if return_inverter:
                        return image, None
                    else:
                        return image
                else:
                    if return_inverter:
                        return image, labels, None
                    else:
                        return image, labels

        else:
            if return_inverter:
                def inverter(labels):
                    return labels

            if labels is None:
                if return_inverter:
                    return image, inverter
                else:
                    return image
            else:
                if return_inverter:
                    return image, labels, inverter
                else:
                    return image, labels

class RandomPatchInf:
    '''
    Randomly samples a patch from an image. The randomness refers to whatever
    randomness may be introduced by the patch coordinate generator, the box filter,
    and the patch validator.

    Input images may be cropped and/or padded along either or both of the two
    spatial dimensions as necessary in order to obtain the required patch.

    This operation is very similar to `RandomPatch`, except that:
    1. This operation runs indefinitely until either a valid patch is found or
       the input image is returned unaltered, i.e. it cannot fail.
    2. If a bound generator is given, a new pair of bounds will be generated
       every `n_trials_max` iterations.
    '''

    def __init__(self,
                 patch_coord_generator,
                 box_filter=None,
                 image_validator=None,
                 bound_generator=None,
                 n_trials_max=50,
                 clip_boxes=True,
                 prob=0.857,
                 background=(0,0,0),
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        '''
        Arguments:
            patch_coord_generator (PatchCoordinateGenerator): A `PatchCoordinateGenerator` object
                to generate the positions and sizes of the patches to be sampled from the input images.
            box_filter (BoxFilter, optional): Only relevant if ground truth bounding boxes are given.
                A `BoxFilter` object to filter out bounding boxes that don't meet the given criteria
                after the transformation. Refer to the `BoxFilter` documentation for details. If `None`,
                the validity of the bounding boxes is not checked.
            image_validator (ImageValidator, optional): Only relevant if ground truth bounding boxes are given.
                An `ImageValidator` object to determine whether a sampled patch is valid. If `None`,
                any outcome is valid.
            bound_generator (BoundGenerator, optional): A `BoundGenerator` object to generate upper and
                lower bound values for the patch validator. Every `n_trials_max` trials, a new pair of
                upper and lower bounds will be generated until a valid patch is found or the original image
                is returned. This bound generator overrides the bound generator of the patch validator.
            n_trials_max (int, optional): Only relevant if ground truth bounding boxes are given.
                The sampler will run indefinitely until either a valid patch is found or the original image
                is returned, but this determines the maxmial number of trials to sample a valid patch for each
                selected pair of lower and upper bounds before a new pair is picked.
            clip_boxes (bool, optional): Only relevant if ground truth bounding boxes are given.
                If `True`, any ground truth bounding boxes will be clipped to lie entirely within the
                sampled patch.
            prob (float, optional): `(1 - prob)` determines the probability with which the original,
                unaltered image is returned.
            background (list/tuple, optional): A 3-tuple specifying the RGB color value of the potential
                background pixels of the scaled images. In the case of single-channel images,
                the first element of `background` will be used as the background pixel value.
            labels_format (dict, optional): A dictionary that defines which index in the last axis of the labels
                of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
        '''

        if not isinstance(patch_coord_generator, PatchCoordinateGenerator):
            raise ValueError("`patch_coord_generator` must be an instance of `PatchCoordinateGenerator`.")
        if not (isinstance(image_validator, ImageValidator) or image_validator is None):
            raise ValueError("`image_validator` must be either `None` or an `ImageValidator` object.")
        if not (isinstance(bound_generator, BoundGenerator) or bound_generator is None):
            raise ValueError("`bound_generator` must be either `None` or a `BoundGenerator` object.")
        self.patch_coord_generator = patch_coord_generator
        self.box_filter = box_filter
        self.image_validator = image_validator
        self.bound_generator = bound_generator
        self.n_trials_max = n_trials_max
        self.clip_boxes = clip_boxes
        self.prob = prob
        self.background = background
        self.labels_format = labels_format
        self.sample_patch = CropPad(patch_ymin=None,
                                    patch_xmin=None,
                                    patch_height=None,
                                    patch_width=None,
                                    clip_boxes=self.clip_boxes,
                                    box_filter=self.box_filter,
                                    background=self.background,
                                    labels_format=self.labels_format)

    def __call__(self, image, labels=None, return_inverter=False):

        img_height, img_width = image.shape[:2]
        self.patch_coord_generator.img_height = img_height
        self.patch_coord_generator.img_width = img_width

        xmin = self.labels_format['xmin']
        ymin = self.labels_format['ymin']
        xmax = self.labels_format['xmax']
        ymax = self.labels_format['ymax']

        # Override the preset labels format.
        if not self.image_validator is None:
            self.image_validator.labels_format = self.labels_format
        self.sample_patch.labels_format = self.labels_format

        while True: # Keep going until we either find a valid patch or return the original image.

            p = np.random.uniform(0,1)
            if p >= (1.0-self.prob):

                # In case we have a bound generator, pick a lower and upper bound for the patch validator.
                if not ((self.image_validator is None) or (self.bound_generator is None)):
                    self.image_validator.bounds = self.bound_generator()

                # Use at most `self.n_trials_max` attempts to find a crop
                # that meets our requirements.
                for _ in range(max(1, self.n_trials_max)):

                    # Generate patch coordinates.
                    patch_ymin, patch_xmin, patch_height, patch_width = self.patch_coord_generator()

                    self.sample_patch.patch_ymin = patch_ymin
                    self.sample_patch.patch_xmin = patch_xmin
                    self.sample_patch.patch_height = patch_height
                    self.sample_patch.patch_width = patch_width

                    # Check if the resulting patch meets the aspect ratio requirements.
                    aspect_ratio = patch_width / patch_height
                    if not (self.patch_coord_generator.min_aspect_ratio <= aspect_ratio <= self.patch_coord_generator.max_aspect_ratio):
                        continue

                    if (labels is None) or (self.image_validator is None):
                        # We either don't have any boxes or if we do, we will accept any outcome as valid.
                        return self.sample_patch(image, labels, return_inverter)
                    else:
                        # Translate the box coordinates to the patch's coordinate system.
                        new_labels = np.copy(labels)
                        new_labels[:, [ymin, ymax]] -= patch_ymin
                        new_labels[:, [xmin, xmax]] -= patch_xmin
                        # Check if the patch contains the minimum number of boxes we require.
                        if self.image_validator(labels=new_labels,
                                                image_height=patch_height,
                                                image_width=patch_width):
                            return self.sample_patch(image, labels, return_inverter)
            else:
                if return_inverter:
                    def inverter(labels):
                        return labels

                if labels is None:
                    if return_inverter:
                        return image, inverter
                    else:
                        return image
                else:
                    if return_inverter:
                        return image, labels, inverter
                    else:
                        return image, labels

class RandomMaxCropFixedAR:
    '''
    Crops the largest possible patch of a given fixed aspect ratio
    from an image.

    Since the aspect ratio of the sampled patches is constant, they
    can subsequently be resized to the same size without distortion.
    '''

    def __init__(self,
                 patch_aspect_ratio,
                 box_filter=None,
                 image_validator=None,
                 n_trials_max=3,
                 clip_boxes=True,
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        '''
        Arguments:
            patch_aspect_ratio (float): The fixed aspect ratio that all sampled patches will have.
            box_filter (BoxFilter, optional): Only relevant if ground truth bounding boxes are given.
                A `BoxFilter` object to filter out bounding boxes that don't meet the given criteria
                after the transformation. Refer to the `BoxFilter` documentation for details. If `None`,
                the validity of the bounding boxes is not checked.
            image_validator (ImageValidator, optional): Only relevant if ground truth bounding boxes are given.
                An `ImageValidator` object to determine whether a sampled patch is valid. If `None`,
                any outcome is valid.
            n_trials_max (int, optional): Only relevant if ground truth bounding boxes are given.
                Determines the maxmial number of trials to sample a valid patch. If no valid patch could
                be sampled in `n_trials_max` trials, returns `None`.
            clip_boxes (bool, optional): Only relevant if ground truth bounding boxes are given.
                If `True`, any ground truth bounding boxes will be clipped to lie entirely within the
                sampled patch.
            labels_format (dict, optional): A dictionary that defines which index in the last axis of the labels
                of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
        '''

        self.patch_aspect_ratio = patch_aspect_ratio
        self.box_filter = box_filter
        self.image_validator = image_validator
        self.n_trials_max = n_trials_max
        self.clip_boxes = clip_boxes
        self.labels_format = labels_format
        self.random_patch = RandomPatch(patch_coord_generator=PatchCoordinateGenerator(), # Just a dummy object
                                        box_filter=self.box_filter,
                                        image_validator=self.image_validator,
                                        n_trials_max=self.n_trials_max,
                                        clip_boxes=self.clip_boxes,
                                        prob=1.0,
                                        can_fail=False,
                                        labels_format=self.labels_format)

    def __call__(self, image, labels=None, return_inverter=False):

        img_height, img_width = image.shape[:2]

        # The ratio of the input image aspect ratio and patch aspect ratio determines the maximal possible crop.
        image_aspect_ratio = img_width / img_height

        if image_aspect_ratio < self.patch_aspect_ratio:
            patch_width = img_width
            patch_height = int(round(patch_width / self.patch_aspect_ratio))
        else:
            patch_height = img_height
            patch_width = int(round(patch_height * self.patch_aspect_ratio))

        # Now that we know the desired height and width for the patch,
        # instantiate an appropriate patch coordinate generator.
        patch_coord_generator = PatchCoordinateGenerator(img_height=img_height,
                                                         img_width=img_width,
                                                         must_match='h_w',
                                                         patch_height=patch_height,
                                                         patch_width=patch_width)

        # The rest of the work is done by `RandomPatch`.
        self.random_patch.patch_coord_generator = patch_coord_generator
        self.random_patch.labels_format = self.labels_format
        return self.random_patch(image, labels, return_inverter)

class RandomPadFixedAR:
    '''
    Adds the minimal possible padding to an image that results in a patch
    of the given fixed aspect ratio that contains the entire image.

    Since the aspect ratio of the resulting images is constant, they
    can subsequently be resized to the same size without distortion.
    '''

    def __init__(self,
                 patch_aspect_ratio,
                 background=(0,0,0),
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        '''
        Arguments:
            patch_aspect_ratio (float): The fixed aspect ratio that all sampled patches will have.
            background (list/tuple, optional): A 3-tuple specifying the RGB color value of the potential
                background pixels of the scaled images. In the case of single-channel images,
                the first element of `background` will be used as the background pixel value.
            labels_format (dict, optional): A dictionary that defines which index in the last axis of the labels
                of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
        '''

        self.patch_aspect_ratio = patch_aspect_ratio
        self.background = background
        self.labels_format = labels_format
        self.random_patch = RandomPatch(patch_coord_generator=PatchCoordinateGenerator(), # Just a dummy object
                                        box_filter=None,
                                        image_validator=None,
                                        n_trials_max=1,
                                        clip_boxes=False,
                                        background=self.background,
                                        prob=1.0,
                                        labels_format=self.labels_format)

    def __call__(self, image, labels=None, return_inverter=False):

        img_height, img_width = image.shape[:2]

        if img_width < img_height:
            patch_height = img_height
            patch_width = int(round(patch_height * self.patch_aspect_ratio))
        else:
            patch_width = img_width
            patch_height = int(round(patch_width / self.patch_aspect_ratio))

        # Now that we know the desired height and width for the patch,
        # instantiate an appropriate patch coordinate generator.
        patch_coord_generator = PatchCoordinateGenerator(img_height=img_height,
                                                         img_width=img_width,
                                                         must_match='h_w',
                                                         patch_height=patch_height,
                                                         patch_width=patch_width)

        # The rest of the work is done by `RandomPatch`.
        self.random_patch.patch_coord_generator = patch_coord_generator
        self.random_patch.labels_format = self.labels_format
        return self.random_patch(image, labels, return_inverter)


class ConvertColor:
    '''
    Converts images between RGB, HSV and grayscale color spaces. This is just a wrapper
    around `cv2.cvtColor()`.
    '''
    def __init__(self, current='RGB', to='HSV', keep_3ch=True):
        '''
        Arguments:
            current (str, optional): The current color space of the images. Can be
                one of 'RGB' and 'HSV'.
            to (str, optional): The target color space of the images. Can be one of
                'RGB', 'HSV', and 'GRAY'.
            keep_3ch (bool, optional): Only relevant if `to == GRAY`.
                If `True`, the resulting grayscale images will have three channels.
        '''
        if not ((current in {'RGB', 'HSV'}) and (to in {'RGB', 'HSV', 'GRAY'})):
            raise NotImplementedError
        self.current = current
        self.to = to
        self.keep_3ch = keep_3ch

    def __call__(self, image, labels=None):
        if self.current == 'RGB' and self.to == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif self.current == 'RGB' and self.to == 'GRAY':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            if self.keep_3ch:
                image = np.stack([image] * 3, axis=-1)
        elif self.current == 'HSV' and self.to == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        elif self.current == 'HSV' and self.to == 'GRAY':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2GRAY)
            if self.keep_3ch:
                image = np.stack([image] * 3, axis=-1)
        if labels is None:
            return image
        else:
            return image, labels

class ConvertDataType:
    '''
    Converts images represented as Numpy arrays between `uint8` and `float32`.
    Serves as a helper for certain photometric distortions. This is just a wrapper
    around `np.ndarray.astype()`.
    '''
    def __init__(self, to='uint8'):
        '''
        Arguments:
            to (string, optional): To which datatype to convert the input images.
                Can be either of 'uint8' and 'float32'.
        '''
        if not (to == 'uint8' or to == 'float32'):
            raise ValueError("`to` can be either of 'uint8' or 'float32'.")
        self.to = to

    def __call__(self, image, labels=None):
        if self.to == 'uint8':
            image = np.round(image, decimals=0).astype(np.uint8)
        else:
            image = image.astype(np.float32)
        if labels is None:
            return image
        else:
            return image, labels

class ConvertTo3Channels:
    '''
    Converts 1-channel and 4-channel images to 3-channel images. Does nothing to images that
    already have 3 channels. In the case of 4-channel images, the fourth channel will be
    discarded.
    '''
    def __init__(self):
        pass

    def __call__(self, image, labels=None):
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3:
            if image.shape[2] == 1:
                image = np.concatenate([image] * 3, axis=-1)
            elif image.shape[2] == 4:
                image = image[:,:,:3]
        if labels is None:
            return image
        else:
            return image, labels

class Hue:
    '''
    Changes the hue of HSV images.

    Important:
        - Expects HSV input.
        - Expects input array to be of `dtype` `float`.
    '''
    def __init__(self, delta):
        '''
        Arguments:
            delta (int): An integer in the closed interval `[-180, 180]` that determines the hue change, where
                a change by integer `delta` means a change by `2 * delta` degrees. Read up on the HSV color format
                if you need more information.
        '''
        if not (-180 <= delta <= 180): raise ValueError("`delta` must be in the closed interval `[-180, 180]`.")
        self.delta = delta

    def __call__(self, image, labels=None):
        image[:, :, 0] = (image[:, :, 0] + self.delta) % 180.0
        if labels is None:
            return image
        else:
            return image, labels

class RandomHue:
    '''
    Randomly changes the hue of HSV images.

    Important:
        - Expects HSV input.
        - Expects input array to be of `dtype` `float`.
    '''
    def __init__(self, max_delta=18, prob=0.5):
        '''
        Arguments:
            max_delta (int): An integer in the closed interval `[0, 180]` that determines the maximal absolute
                hue change.
            prob (float, optional): `(1 - prob)` determines the probability with which the original,
                unaltered image is returned.
        '''
        if not (0 <= max_delta <= 180): raise ValueError("`max_delta` must be in the closed interval `[0, 180]`.")
        self.max_delta = max_delta
        self.prob = prob
        self.change_hue = Hue(delta=0)

    def __call__(self, image, labels=None):
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            self.change_hue.delta = np.random.uniform(-self.max_delta, self.max_delta)
            return self.change_hue(image, labels)
        elif labels is None:
            return image
        else:
            return image, labels

class Saturation:
    '''
    Changes the saturation of HSV images.

    Important:
        - Expects HSV input.
        - Expects input array to be of `dtype` `float`.
    '''
    def __init__(self, factor):
        '''
        Arguments:
            factor (float): A float greater than zero that determines saturation change, where
                values less than one result in less saturation and values greater than one result
                in more saturation.
        '''
        if factor <= 0.0: raise ValueError("It must be `factor > 0`.")
        self.factor = factor

    def __call__(self, image, labels=None):
        image[:,:,1] = np.clip(image[:,:,1] * self.factor, 0, 255)
        if labels is None:
            return image
        else:
            return image, labels

class RandomSaturation:
    '''
    Randomly changes the saturation of HSV images.

    Important:
        - Expects HSV input.
        - Expects input array to be of `dtype` `float`.
    '''
    def __init__(self, lower=0.3, upper=2.0, prob=0.5):
        '''
        Arguments:
            lower (float, optional): A float greater than zero, the lower bound for the random
                saturation change.
            upper (float, optional): A float greater than zero, the upper bound for the random
                saturation change. Must be greater than `lower`.
            prob (float, optional): `(1 - prob)` determines the probability with which the original,
                unaltered image is returned.
        '''
        if lower >= upper: raise ValueError("`upper` must be greater than `lower`.")
        self.lower = lower
        self.upper = upper
        self.prob = prob
        self.change_saturation = Saturation(factor=1.0)

    def __call__(self, image, labels=None):
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            self.change_saturation.factor = np.random.uniform(self.lower, self.upper)
            return self.change_saturation(image, labels)
        elif labels is None:
            return image
        else:
            return image, labels

class Brightness:
    '''
    Changes the brightness of RGB images.

    Important:
        - Expects RGB input.
        - Expects input array to be of `dtype` `float`.
    '''
    def __init__(self, delta):
        '''
        Arguments:
            delta (int): An integer, the amount to add to or subtract from the intensity
                of every pixel.
        '''
        self.delta = delta

    def __call__(self, image, labels=None):
        image = np.clip(image + self.delta, 0, 255)
        if labels is None:
            return image
        else:
            return image, labels

class RandomBrightness:
    '''
    Randomly changes the brightness of RGB images.

    Important:
        - Expects RGB input.
        - Expects input array to be of `dtype` `float`.
    '''
    def __init__(self, lower=-84, upper=84, prob=0.5):
        '''
        Arguments:
            lower (int, optional): An integer, the lower bound for the random brightness change.
            upper (int, optional): An integer, the upper bound for the random brightness change.
                Must be greater than `lower`.
            prob (float, optional): `(1 - prob)` determines the probability with which the original,
                unaltered image is returned.
        '''
        if lower >= upper: raise ValueError("`upper` must be greater than `lower`.")
        self.lower = float(lower)
        self.upper = float(upper)
        self.prob = prob
        self.change_brightness = Brightness(delta=0)

    def __call__(self, image, labels=None):
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            self.change_brightness.delta = np.random.uniform(self.lower, self.upper)
            return self.change_brightness(image, labels)
        elif labels is None:
            return image
        else:
            return image, labels

class Contrast:
    '''
    Changes the contrast of RGB images.

    Important:
        - Expects RGB input.
        - Expects input array to be of `dtype` `float`.
    '''
    def __init__(self, factor):
        '''
        Arguments:
            factor (float): A float greater than zero that determines contrast change, where
                values less than one result in less contrast and values greater than one result
                in more contrast.
        '''
        if factor <= 0.0: raise ValueError("It must be `factor > 0`.")
        self.factor = factor

    def __call__(self, image, labels=None):
        image = np.clip(127.5 + self.factor * (image - 127.5), 0, 255)
        if labels is None:
            return image
        else:
            return image, labels

class RandomContrast:
    '''
    Randomly changes the contrast of RGB images.

    Important:
        - Expects RGB input.
        - Expects input array to be of `dtype` `float`.
    '''
    def __init__(self, lower=0.5, upper=1.5, prob=0.5):
        '''
        Arguments:
            lower (float, optional): A float greater than zero, the lower bound for the random
                contrast change.
            upper (float, optional): A float greater than zero, the upper bound for the random
                contrast change. Must be greater than `lower`.
            prob (float, optional): `(1 - prob)` determines the probability with which the original,
                unaltered image is returned.
        '''
        if lower >= upper: raise ValueError("`upper` must be greater than `lower`.")
        self.lower = lower
        self.upper = upper
        self.prob = prob
        self.change_contrast = Contrast(factor=1.0)

    def __call__(self, image, labels=None):
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            self.change_contrast.factor = np.random.uniform(self.lower, self.upper)
            return self.change_contrast(image, labels)
        elif labels is None:
            return image
        else:
            return image, labels

class Gamma:
    '''
    Changes the gamma value of RGB images.

    Important: Expects RGB input.
    '''
    def __init__(self, gamma):
        '''
        Arguments:
            gamma (float): A float greater than zero that determines gamma change.
        '''
        if gamma <= 0.0: raise ValueError("It must be `gamma > 0`.")
        self.gamma = gamma
        self.gamma_inv = 1.0 / gamma
        # Build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values.
        self.table = np.array([((i / 255.0) ** self.gamma_inv) * 255 for i in np.arange(0, 256)]).astype("uint8")

    def __call__(self, image, labels=None):
        image = cv2.LUT(image, table)
        if labels is None:
            return image
        else:
            return image, labels

class RandomGamma:
    '''
    Randomly changes the gamma value of RGB images.

    Important: Expects RGB input.
    '''
    def __init__(self, lower=0.25, upper=2.0, prob=0.5):
        '''
        Arguments:
            lower (float, optional): A float greater than zero, the lower bound for the random
                gamma change.
            upper (float, optional): A float greater than zero, the upper bound for the random
                gamma change. Must be greater than `lower`.
            prob (float, optional): `(1 - prob)` determines the probability with which the original,
                unaltered image is returned.
        '''
        if lower >= upper: raise ValueError("`upper` must be greater than `lower`.")
        self.lower = lower
        self.upper = upper
        self.prob = prob

    def __call__(self, image, labels=None):
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            gamma = np.random.uniform(self.lower, self.upper)
            change_gamma = Gamma(gamma=gamma)
            return change_gamma(image, labels)
        elif labels is None:
            return image
        else:
            return image, labels

class HistogramEqualization:
    '''
    Performs histogram equalization on HSV images.

    Importat: Expects HSV input.
    '''
    def __init__(self):
        pass

    def __call__(self, image, labels=None):
        image[:,:,2] = cv2.equalizeHist(image[:,:,2])
        if labels is None:
            return image
        else:
            return image, labels

class RandomHistogramEqualization:
    '''
    Randomly performs histogram equalization on HSV images. The randomness only refers
    to whether or not the equalization is performed.

    Importat: Expects HSV input.
    '''
    def __init__(self, prob=0.5):
        '''
        Arguments:
            prob (float, optional): `(1 - prob)` determines the probability with which the original,
                unaltered image is returned.
        '''
        self.prob = prob
        self.equalize = HistogramEqualization()

    def __call__(self, image, labels=None):
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            return self.equalize(image, labels)
        elif labels is None:
            return image
        else:
            return image, labels

class ChannelSwap:
    '''
    Swaps the channels of images.
    '''
    def __init__(self, order):
        '''
        Arguments:
            order (tuple): A tuple of integers that defines the desired channel order
                of the input images after the channel swap.
        '''
        self.order = order

    def __call__(self, image, labels=None):
        image = image[:,:,self.order]
        if labels is None:
            return image
        else:
            return image, labels

class RandomChannelSwap:
    '''
    Randomly swaps the channels of RGB images.

    Important: Expects RGB input.
    '''
    def __init__(self, prob=0.5):
        '''
        Arguments:
            prob (float, optional): `(1 - prob)` determines the probability with which the original,
                unaltered image is returned.
        '''
        self.prob = prob
        # All possible permutations of the three image channels except the original order.
        self.permutations = ((0, 2, 1),
                             (1, 0, 2), (1, 2, 0),
                             (2, 0, 1), (2, 1, 0))
        self.swap_channels = ChannelSwap(order=(0, 1, 2))

    def __call__(self, image, labels=None):
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            i = np.random.randint(5) # There are 6 possible permutations.
            self.swap_channels.order = self.permutations[i]
            return self.swap_channels(image, labels)
        elif labels is None:
            return image
        else:
            return image, labels




class SSDRandomCrop:
    '''
    Performs the same random crops as defined by the `batch_sampler` instructions
    of the original Caffe implementation of SSD. A description of this random cropping
    strategy can also be found in the data augmentation section of the paper:
    https://arxiv.org/abs/1512.02325
    '''

    def __init__(self, labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        '''
        Arguments:
            labels_format (dict, optional): A dictionary that defines which index in the last axis of the labels
                of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
        '''

        self.labels_format = labels_format

        # This randomly samples one of the lower IoU bounds defined
        # by the `sample_space` every time it is called.
        self.bound_generator = BoundGenerator(sample_space=((None, None),
                                                            (0.1, None),
                                                            (0.3, None),
                                                            (0.5, None),
                                                            (0.7, None),
                                                            (0.9, None)),
                                              weights=None)

        # Produces coordinates for candidate patches such that the height
        # and width of the patches are between 0.3 and 1.0 of the height
        # and width of the respective image and the aspect ratio of the
        # patches is between 0.5 and 2.0.
        self.patch_coord_generator = PatchCoordinateGenerator(must_match='h_w',
                                                              min_scale=0.3,
                                                              max_scale=1.0,
                                                              scale_uniformly=False,
                                                              min_aspect_ratio = 0.5,
                                                              max_aspect_ratio = 2.0)

        # Filters out boxes whose center point does not lie within the
        # chosen patches.
        self.box_filter = BoxFilter(check_overlap=True,
                                    check_min_area=False,
                                    check_degenerate=False,
                                    overlap_criterion='center_point',
                                    labels_format=self.labels_format)

        # Determines whether a given patch is considered a valid patch.
        # Defines a patch to be valid if at least one ground truth bounding box
        # (n_boxes_min == 1) has an IoU overlap with the patch that
        # meets the requirements defined by `bound_generator`.
        self.image_validator = ImageValidator(overlap_criterion='iou',
                                              n_boxes_min=1,
                                              labels_format=self.labels_format,
                                              border_pixels='half')

        # Performs crops according to the parameters set in the objects above.
        # Runs until either a valid patch is found or the original input image
        # is returned unaltered. Runs a maximum of 50 trials to find a valid
        # patch for each new sampled IoU threshold. Every 50 trials, the original
        # image is returned as is with probability (1 - prob) = 0.143.
        self.random_crop = RandomPatchInf(patch_coord_generator=self.patch_coord_generator,
                                          box_filter=self.box_filter,
                                          image_validator=self.image_validator,
                                          bound_generator=self.bound_generator,
                                          n_trials_max=50,
                                          clip_boxes=True,
                                          prob=0.857,
                                          labels_format=self.labels_format)

    def __call__(self, image, labels=None, return_inverter=False):
        self.random_crop.labels_format = self.labels_format
        return self.random_crop(image, labels, return_inverter)

class SSDExpand:
    '''
    Performs the random image expansion as defined by the `train_transform_param` instructions
    of the original Caffe implementation of SSD. A description of this expansion strategy
    can also be found in section 3.6 ("Data Augmentation for Small Object Accuracy") of the paper:
    https://arxiv.org/abs/1512.02325
    '''

    def __init__(self, background=(123, 117, 104), labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        '''
        Arguments:
            background (list/tuple, optional): A 3-tuple specifying the RGB color value of the
                background pixels of the translated images.
            labels_format (dict, optional): A dictionary that defines which index in the last axis of the labels
                of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
        '''

        self.labels_format = labels_format

        # Generate coordinates for patches that are between 1.0 and 4.0 times
        # the size of the input image in both spatial dimensions.
        self.patch_coord_generator = PatchCoordinateGenerator(must_match='h_w',
                                                              min_scale=1.0,
                                                              max_scale=4.0,
                                                              scale_uniformly=True)

        # With probability 0.5, place the input image randomly on a canvas filled with
        # mean color values according to the parameters set above. With probability 0.5,
        # return the input image unaltered.
        self.expand = RandomPatch(patch_coord_generator=self.patch_coord_generator,
                                  box_filter=None,
                                  image_validator=None,
                                  n_trials_max=1,
                                  clip_boxes=False,
                                  prob=0.5,
                                  background=background,
                                  labels_format=self.labels_format)

    def __call__(self, image, labels=None, return_inverter=False):
        self.expand.labels_format = self.labels_format
        return self.expand(image, labels, return_inverter)

class SSDPhotometricDistortions:
    '''
    Performs the photometric distortions defined by the `train_transform_param` instructions
    of the original Caffe implementation of SSD.
    '''

    def __init__(self):

        self.convert_RGB_to_HSV = ConvertColor(current='RGB', to='HSV')
        self.convert_HSV_to_RGB = ConvertColor(current='HSV', to='RGB')
        self.convert_to_float32 = ConvertDataType(to='float32')
        self.convert_to_uint8 = ConvertDataType(to='uint8')
        self.convert_to_3_channels = ConvertTo3Channels()
        self.random_brightness = RandomBrightness(lower=-32, upper=32, prob=0.5)
        self.random_contrast = RandomContrast(lower=0.5, upper=1.5, prob=0.5)
        self.random_saturation = RandomSaturation(lower=0.5, upper=1.5, prob=0.5)
        self.random_hue = RandomHue(max_delta=18, prob=0.5)
        self.random_channel_swap = RandomChannelSwap(prob=0.0)

        self.sequence1 = [self.convert_to_3_channels,
                          self.convert_to_float32,
                          self.random_brightness,
                          self.random_contrast,
                          self.convert_to_uint8,
                          self.convert_RGB_to_HSV,
                          self.convert_to_float32,
                          self.random_saturation,
                          self.random_hue,
                          self.convert_to_uint8,
                          self.convert_HSV_to_RGB,
                          self.random_channel_swap]

        self.sequence2 = [self.convert_to_3_channels,
                          self.convert_to_float32,
                          self.random_brightness,
                          self.convert_to_uint8,
                          self.convert_RGB_to_HSV,
                          self.convert_to_float32,
                          self.random_saturation,
                          self.random_hue,
                          self.convert_to_uint8,
                          self.convert_HSV_to_RGB,
                          self.convert_to_float32,
                          self.random_contrast,
                          self.convert_to_uint8,
                          self.random_channel_swap]

    def __call__(self, image, labels):

        # Choose sequence 1 with probability 0.5.
        if np.random.choice(2):

            for transform in self.sequence1:
                image, labels = transform(image, labels)
            return image, labels
        # Choose sequence 2 with probability 0.5.
        else:

            for transform in self.sequence2:
                image, labels = transform(image, labels)
            return image, labels

class SSDDataAugmentation:
    '''
    Reproduces the data augmentation pipeline used in the training of the original
    Caffe implementation of SSD.
    '''

    def __init__(self,
                 img_height=300,
                 img_width=300,
                 background=(123, 117, 104),
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        '''
        Arguments:
            height (int): The desired height of the output images in pixels.
            width (int): The desired width of the output images in pixels.
            background (list/tuple, optional): A 3-tuple specifying the RGB color value of the
                background pixels of the translated images.
            labels_format (dict, optional): A dictionary that defines which index in the last axis of the labels
                of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
        '''

        self.labels_format = labels_format

        self.photometric_distortions = SSDPhotometricDistortions()
        self.expand = SSDExpand(background=background, labels_format=self.labels_format)
        self.random_crop = SSDRandomCrop(labels_format=self.labels_format)
        self.random_flip = RandomFlip(dim='horizontal', prob=0.5, labels_format=self.labels_format)

        # This box filter makes sure that the resized images don't contain any degenerate boxes.
        # Resizing the images could lead the boxes to becomes smaller. For boxes that are already
        # pretty small, that might result in boxes with height and/or width zero, which we obviously
        # cannot allow.
        self.box_filter = BoxFilter(check_overlap=False,
                                    check_min_area=False,
                                    check_degenerate=True,
                                    labels_format=self.labels_format)

        self.resize = ResizeRandomInterp(height=img_height,
                                         width=img_width,
                                         interpolation_modes=[cv2.INTER_NEAREST,
                                                              cv2.INTER_LINEAR,
                                                              cv2.INTER_CUBIC,
                                                              cv2.INTER_AREA,
                                                              cv2.INTER_LANCZOS4],
                                         box_filter=self.box_filter,
                                         labels_format=self.labels_format)

        self.sequence = [self.photometric_distortions,
                         self.expand,
                         self.random_crop,
                         self.random_flip,
                         self.resize]

    def __call__(self, image, labels, return_inverter=False):
        self.expand.labels_format = self.labels_format
        self.random_crop.labels_format = self.labels_format
        self.random_flip.labels_format = self.labels_format
        self.resize.labels_format = self.labels_format

        inverters = []

        for transform in self.sequence:
            if return_inverter and ('return_inverter' in inspect.signature(transform).parameters):
                image, labels, inverter = transform(image, labels, return_inverter=True)
                inverters.append(inverter)
            else:
                image, labels = transform(image, labels)

        if return_inverter:
            return image, labels, inverters[::-1]
        else:
            return image, labels
