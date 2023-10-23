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
    runner = OnnxTestRunner("demo1", "/root/Workspace/config/llama_config.toml")
    # model_file = r'/data/huochenghai/onnx_model/shufflenet-9.onnx'
    # model_file = '/compiler/huochenghai/GNNE/nncase_demo/examples/release_isp_object_detect_nncase/data/yolov5sFocus_320x3.onnx'
    # model_file = '/data/huochenghai/GNNE/nncase_demo/examples/release_isp_retinaface_mb_320_nncase/data/retinaface_mobile0.25_320.onnx'
    # model_file = '/data/huochenghai/GNNE/nncase_demo/examples/release_isp_face_landmarks106_nncase/data/retinaface_mobile0.25_320.onnx'
    # model_file = '/data/huochenghai/GNNE/nncase_demo/examples/release_isp_face_landmarks106_nncase/data/v3.onnx'
    # model_file = '/data/huochenghai/fixed_input.onnx'
    # model_file = '/data/huochenghai/GNNE/nncase_demo/examples/release_isp_face_alignment_from_box_nncase/data/mb1_120x120.onnx'
    # model_file = '/data/huochenghai/GNNE/nncase_demo/examples/release_isp_face_recog_mbface_nncase/data/mbface.onnx'
    # model_file = '/data/huochenghai/GNNE/k510-gnne-compiler-tests/zhoumeng-model/resnet50v1/model_f32.onnx'
    # model_file = '/data/huochenghai/deploy_modify.onnx'
    # model_file = '/data/huochenghai/nanodet_mobilenetv2_416.onnx'
    # model_file = '/data/huochenghai/yolov5_face_n0.5_256x256.onnx'
    # model_file = '/data/huochenghai/yolov5s_0.5_640_dropact.onnx'
    # model_file = '/data/huochenghai/GNNE/nncase_demo/examples/release_isp_object_detect_nncase/data/yolov5sFocus_320x3.onnx'
    # model_file = '/data/huochenghai/nanodet_yolov5s_0.5_head_nospp_640.onnx'
    # model_file = '/data/huochenghai/dw_21x21_model.onnx'
    # model_file = '/compiler/huochenghai/GNNE/nncase/tests_output/test_decoder_part/simplified.onnx'
    # model_file = '/data/huochenghai/onnx_model/yolop_self.onnx'
    # model_file = '/data/huochenghai/yolov5s_640x640_sigmoid_weights.onnx'
    # model_file = '/data/huochenghai/models/yolov5s_640_sigmoid.onnx'
    # model_file = '/data/huochenghai/best_batchsize16_300'
    # model_file = '/data/huochenghai/candy-9.onnx'
    # model_file = '/data/huochenghai/glint360k_cosface_r18_fp16_0.1.onnx'
    # model_file = '/data/huochenghai/cls_fixed2.onnx'
    # model_file = '/data/huochenghai/stereo_ranpara.onnx'
    # model_file = '/data/huochenghai/stereoNet.onnx'
    # model_file = '/data/huochenghai/deploy_modify.onnx'
    # model_file = '/data/huochenghai/model.onnx'
    # model_file = "/data/huochenghai/onnx_model/lite-transformer-encoder.onnx"
    # model_file = '/data/huochenghai/onnx_model/lite-transformer-decoder.onnx'
    # model_file = '/data/huochenghai/pose_vgg_half_030.onnx'
    # model_file = '/data/huochenghai/pose1040.onnx'
    # model_file = '/data/huochenghai/net.onnx'
    # model_file = '/data/huochenghai/face_expression.onnx'
    # model_file = '/data/huochenghai/model_fixed_input_size.onnx'
    # model_file = '/data/huochenghai/model_none_lstm.onnx'
    # model_file = '/data/huochenghai/squeezenet1_1.onnx'
    # model_file = '/data/huochenghai/resnet_tom.onnx'
    # model_file = '/data/huochenghai/Ultralight-Nano-SimplePose.onnx'
    # model_file = "/data/huochenghai/yolov5sface_640x640_6output.onnx"
    # model_file = "/data/huochenghai/model-y1.onnx"
    # model_file = "/data/huochenghai/sim_5.onnx"
    # model_file = "/data/huochenghai/person_yolov5s_0.5_nospp_640_nncase.onnx"
    # model_file = "/data/huochenghai/rec_2_layer_lstm.onnx"
    # model_file = "/compiler/huochenghai/east_128_640.onnx"
    # model_file = "/compiler/huochenghai/GNNE/k230-gnne-compiler-tests/benchmark-test/CRNN/ocr_rec_model_32-608.onnx"
    # model_file = "/data/huochenghai/scrfd_person_2.5g_fixed_input_size_simplify.onnx"
    # model_file = "/data/huochenghai/models/model_128-640-11.onnx"
    # model_file = "/data/huochenghai/GNNE/nncase/tests_output/simplified.onnx"
    # model_file = "/data/huochenghai/dw_deconv.onnx"
    # model_file = "/data/huochenghai/GNNE/nncase/tests_output/test_exchannel_rhs_shape0-lhs_shape0_/simplified.onnx"
    # model_file = "/data/huochenghai/GNNE/k230-gnne-compiler-tests/benchmark-test/lite-transformer/lite_transformer_encoder_L10.onnx"
    # model_file = "/data/huochenghai/GNNE/k230-gnne-compiler-tests/benchmark-test/lite-transformer/lite_transformer_decoder_L10.onnx"
    # model_file = "/data/huochenghai/lite_transformer_decoder_L10.onnx"
    # model_file = "/data/huochenghai/GNNE/k230-gnne-compiler-tests/benchmark-test/yolov5s/yolov5s_640_sigmoid.onnx"
    # model_file = "/data/huochenghai/GNNE/k230-gnne-compiler-tests/benchmark-test/efficientnet/efficientnet_b0_224x224.onnx"
    # model_file = "/data/huochenghai/GNNE/k230-gnne-compiler-tests/benchmark-test/mobile-facenet/mbface_sim_224.onnx"
    # model_file = "/compiler/huochenghai/GNNE/k230-gnne-compiler-tests/benchmark-test/mobile-retinaface/retinaface_mobile0.25_320_simplified.onnx"
    # model_file = "/data/huochenghai/GNNE/k230-gnne-compiler-tests/benchmark-test/mobilenet-v1-ssd/ssd_mobilenetv1_300x300.onnx"
    # model_file = "/data/huochenghai/GNNE/k230-gnne-compiler-tests/benchmark-test/mobilenetv2-yolov3/yolov3_mobilenetv2_no_postprocess.onnx"
    # model_file = "/data/huochenghai/GNNE/k230-gnne-compiler-tests/benchmark-test/mobilenet-v2-ssd/ssd_mobilenetv2_300x300.onnx"
    # model_file = "/data/huochenghai/GNNE/k230-gnne-compiler-tests/benchmark-test/yolov5m/yolov5_m_320x320_with_sigmoid.onnx"
    # model_file = "/compiler/huochenghai/GNNE/k230-gnne-compiler-tests/benchmark-test/yolov5s_face/yolov5sface_640x640_6output.onnx"
    # model_file = "/data/huochenghai/GNNE/k230-gnne-compiler-tests/benchmark-test/yolox/yolox_s.onnx"
    # model_file = "/data/huochenghai/GNNE/k230-gnne-compiler-tests/benchmark-test/Ultralight-SimplePose/Ultralight-SimplePose.onnx"
    # model_file = "/compiler/huochenghai/GNNE/k230-gnne-compiler-tests/benchmark-test/reid/osnet_x1_0.onnx"
    # model_file = "/compiler/huochenghai/GNNE/k230-gnne-compiler-tests/benchmark-test/yolov7/0_yolov7-tiny-silu_320x320.onnx"
    # model_file = "/compiler/huochenghai/GNNE/k230-gnne-compiler-tests/benchmark-test/reid/osnet_ain_x1_0.onnx"
    # model_file = "/data/huochenghai/GNNE/k230-gnne-compiler-tests/benchmark-test/reid/osnet_ibn_x1_0.onnx"
    # model_file = "/compiler/huochenghai/GNNE/k230-gnne-compiler-tests/benchmark-test/wzm/wzm_stereo6g.onnx"
    # model_file = "/compiler/huochenghai/GNNE/k230-gnne-compiler-tests/benchmark-test/wzm/wzm_stereo.onnx"
    # model_file = "/data/huochenghai/GNNE/nncase/tests_output/test_matmul_constant-in_shape0_/simplified.onnx"
    # model_file = "/compiler/huochenghai/GNNE/k230-gnne-compiler-tests/benchmark-test/lite-transformer/youdaonmt/encoder_model.onnx"
    # model_file = "/compiler/huochenghai/GNNE/k230-gnne-compiler-tests/benchmark-test/lite-transformer/youdaonmt/decoder_model.onnx"
    # model_file = "/compiler/huochenghai/GNNE/k230-gnne-compiler-tests/benchmark-test/resnetv1_50/onnx/resnet50v1.onnx"
    # model_file = "/compiler/huochenghai/GNNE/k230-gnne-ccompiler-tests/benchmark-test/lite-transformer/lite_transformer_encoder_L10.onnx"
    # model_file = "/compiler/huochenghai/can3_10.0s_20221011084724.onnx"
    # model_file = "/compiler/huochenghai/lstm_256.onnx"
    # model_file = "/compiler/huochenghai/weilai/simplified_det.onnx"
    # model_file = "/compiler/huochenghai/models/daniu_nmt_enc.onnx"
    # model_file = "/compiler/huochenghai/GNNE/k230-gnne-compiler-tests/benchmark-test/centersnap/CenterSnap.onnx"
    # model_file = "/compiler/huochenghai/GNNE/nncase/tests_output/daniu_enc.onnx"
    # model_file = "/compiler/huochenghai/GNNE/k230-gnne-compiler-tests/benchmark-test/yolop/yolop_self.onnx"
    # model_file = "/compiler/huochenghai/GNNE/k230-gnne-compiler-tests/benchmark-test/daniu/e2z/dec.onnx"
    # model_file = "/compiler/huochenghai/GNNE/k230-gnne-compiler-tests/benchmark-test/daniu/z2e/enc.onnx"
    # model_file = "/compiler/huochenghai/GNNE/k230-gnne-compiler-tests/benchmark-test/daniu/TTS/zho/fix.onnx"
    # model_file = "/compiler/huochenghai/GNNE/k230-gnne-compiler-tests/benchmark-test/CRNN/ocr_rec_model_32-608.onnx"
    # model_file = "/compiler/huochenghai/GNNE/nncase/tests_output/crnn_part.onnx"
    # model_file = "/compiler/huochenghai/GNNE/k230-gnne-compiler-tests/benchmark-test/mobile-facenet/mbface_sim_224.onnx"
    # model_file = "/compiler/huochenghai/GNNE/k230-gnne-compiler-tests/FasterTransformer/LongFormer/longformer-base-4096.onnx"
    # model_file = '/data/huochenghai/GNNE/k230-gnne-compiler-tests/StableDiffusion/onnx-stable-diffusion-v1-5/vae_decoder/model.onnx'
    # model_file = "/data/huochenghai/llama_scrach/65B/decoder-merge-0.onnx"
    # model_file = "/root/Downloads/decoder-merge-0.onnx"
    model_file = "/root/Downloads/64B-4-layers/decoder-merge-all.onnx"

    # runner.set_shape_var({"batch_size": 1, "num_channels_latent": 4, "height_latent": 64, "width_latent": 64})
    runner.set_shape_var({"N": 384})
    runner.run(model_file)


if __name__ == "__main__":
    pytest.main(
        ['-vvs', __file__])