# Copyright 2017 Weijing Shi. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
""" This file contains functions to normalize images. """


import cv2
import numpy as np

METHOD_MAP = {
'hist': equalizeHist
}

def equalizeHist(image):
    """ histogram equalization using openCV.
    Args:
        image: OpenCV image mat or numpy 3D array.
    Returns:
        equalized_image: OpenCV image mat.
    """
    return cv2.equalizeHist(image)

#TODO google the meaning of the algorithm
def clahe(image, clipLimit=2.0, tileGridSize=(8,8)):
    """ Contrast Limited Adaptive Histogram Equalization using openCV.
    Args:
        image: OpenCV image mat or numpy 3D array.
    Returns:
        equalized_image: OpenCV image mat.
    """
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(image)

def localNormalize(image, kernel_size=(10,10)):
    """ Take the local area (kernel_size) of image and normalize it.
    Args:
        image: OpenCV image mat or numpy 3D array.
        kernel_size: the size of local area to normalize.
    Returns:
        locally_normalized_image: numpy 3D array.
    """
    square_image = np.sqare(np.float32(image))
    local_sum_square = cv2.filter2D(square_image, -1, np.ones(kernel_size))
    return np.uint8(image/local_sum_square)
