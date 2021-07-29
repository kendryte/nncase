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
"""System test: test demo"""
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel

import pytest
from tflite_test_runner import TfliteTestRunner


def test_demo_tf(request):
    runner = TfliteTestRunner(request.node.name, ['k510'])
    # model_file = '/data/huochenghai/GNNE/k510-gnne-compiler-tests/golden-model/deeplabv3_plus_mobilenet_v2_fix_dilations/tflite/model_f32.tflite'
    # model_file = '/data/huochenghai/GNNE/k510-gnne-compiler-tests/golden-model/inceptionv4/tflite/model_f32.tflite'
    # model_file = '/data/huochenghai/GNNE/k510-gnne-compiler-tests/golden-model/yolov3/model_f32.tflite'
    # model_file = '/data/huochenghai/GNNE/k510-gnne-compiler-tests/golden-model/deeplabv3_plus_mobilenet_v2_fix_dilations/tflite/model_f32.tflite'
    # model_file = '/data/huochenghai/GNNE/nncase_demo/examples/release_isp_face_recog_from_k210_nncase/data/fd/model_f32.tflite'
    # model_file = '/data/huochenghai/GNNE/k510-gnne-compiler-tests/mini-test/conv_base_case/conv_base_case_108/model_f32.tflite'
    # model_file = '/data/huochenghai/GNNE/nncase_demo/examples/release_isp_face_recog_from_k210_nncase/data/fe/model_f32.tflite'
    model_file = '/data/huochenghai/GNNE/nncase_demo/examples/release_isp_person_pose_nncase/model_f32.tflite'
    # model_file = '/data/huochenghai/GNNE/nncase_demo/examples/release_img_hand_pose_by_mediapipe_nncase/data/hd/model_f32.tflite'
    # model_file = '/data/huochenghai/GNNE/nncase_demo/examples/release_img_hand_pose_by_mediapipe/data/hlm/model_f32.tflite'
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(['-vv', '/data/huochenghai/GNNE/nncase/tests/importer/tflite/model/test_demo_tf.py'])
