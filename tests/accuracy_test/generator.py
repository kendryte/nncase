# Copyright 2019-2021 Canaan Inc.
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
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel

from typing import Any, Dict, List, Tuple, Union
import numpy as np
import os
import cv2


class Generator:

    def from_random(self, shape: List[int], dtype: np.dtype, abs: bool = False) -> np.ndarray:
        if dtype == np.uint8:
            data = np.random.randint(0, 256, shape)
        elif dtype == np.int8:
            data = np.random.randint(-128, 128, shape)
        elif dtype == bool:
            data = np.random.rand(*shape) > 0.5
        elif dtype == np.int32:
            data = np.random.randint(1, 5, size=shape, dtype='int32')
        elif dtype == np.int64:
            data = np.random.randint(1, 5, size=shape, dtype='int64')
            # data = np.random.randint(1, 128, size=shape, dtype='int64')
        else:
            data = np.random.rand(*shape)
        data = data.astype(dtype=dtype)

        if abs:
            return np.abs(data)
        return data

    def from_bin(self, shape: List[int], dtype: np.dtype, bin_file: str):
        '''read data, file name with increase index and data type. e.g. calib_0_0.bin, calib_0_1.bin
        [generator.inputs]
        method = 'bin'

        [generator.inputs.bin]
        args = '/mnt/nncase/tests_output/test_mobile_retinaface_preprocess_3/input'

        [generator.calibs]
        method = 'bin'

        [generator.calibs.bin]
        args = '/mnt/nncase/tests_output/test_mobile_retinaface_preprocess_3/calib'
        '''

        data = np.fromfile(bin_file, dtype)
        return np.reshape(data, shape)

    def from_image(self, shape: List[int], dtype: np.dtype, img_file: str) -> np.ndarray:
        """ read rgb image , return the preprocessed image.
        [generator.inputs]
        method = 'image'

        [generator.inputs.image]
        args = '/mnt/mb_retinaface_input/'

        [generator.calibs]
        method = 'image'

        [generator.calibs.image]
        args = '/mnt/calibration_dataset/'
        """

        transpose = True

        if shape[1] == 3:
            h = shape[2]
            w = shape[3]
            transpose = True
        elif shape[3] == 3:
            h = shape[1]
            w = shape[2]
            transpose = False

        padded_img = np.ones((h, w, 3), dtype=np.uint8) * 114
        img = cv2.imread(img_file)

        # resize
        r = min(h / img.shape[0], w / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)

        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

        # bgr2rgb
        padded_img = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)

        # transpose
        if transpose:
            padded_img = padded_img.transpose((2, 0, 1))

        padded_img = np.ascontiguousarray(padded_img)
        padded_img = np.reshape(padded_img, shape)
        padded_img = padded_img.astype(dtype=dtype)
        return padded_img

    def from_constant_of_shape(self, shape: List[int], dtype: np.dtype) -> np.ndarray:
        return np.array(shape, dtype=dtype)
