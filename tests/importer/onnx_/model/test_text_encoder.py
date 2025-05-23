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

# from lzma import MODE_FAST
# from xml.parsers.expat import model
import pytest
from onnx_test_runner import OnnxTestRunner


def test_demo(request):
    # runner = OnnxTestRunner(request.node.name, "/root/Workspace/nncase/tests/importer/onnx_/model/llama_config.toml")
    runner = OnnxTestRunner("text_encoder", "/root/Workspace/config/text_config.toml")
    # runner = OnnxTestRunner("text_encoder")
    #
    model_file = "/root/Downloads/Models/text_encoder_model.onnx"

    # runner.set_shape_var({"batch_size": 1, "num_channels_latent": 4, "height_latent": 64, "width_latent": 64})
    # runner.set_shape_var({"N": 384})
    runner.set_shape_var({"batch_size": 1, "sequence_length": 77})
    # runner.set_shape_var({"batch_size:1", "sequence_length:77"})
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(
        ['-vvs', __file__])
