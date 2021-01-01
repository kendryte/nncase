/* Copyright 2020 Canaan Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "../tflite_importer.h"
#include <nncase/ir/ops/resize_image.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

DEFINE_TFLITE_LOWER(RESIZE_BILINEAR)
{
    convert_resize_image(op, image_resize_bilinear);
}

DEFINE_TFLITE_LOWER(RESIZE_NEAREST_NEIGHBOR)
{
    convert_resize_image(op, image_resize_nearest_neighbor);
}

void tflite_importer::convert_resize_image(const tflite::Operator &op, image_resize_mode_t mode)
{
    auto &input = get_tensor(op.inputs(), 0);
    auto &output = get_tensor(op.outputs(), 0);
    auto new_size_tensor = load_tensor<int32_t, 1>(get_tensor(op.inputs(), 1));
    std::array<int32_t, 2> new_size{ new_size_tensor[0], new_size_tensor[1] };

    auto align_corners = op.builtin_options_type() == tflite::BuiltinOptions_ResizeBilinearOptions
        ? op.builtin_options_as_ResizeBilinearOptions()->align_corners()
        : op.builtin_options_as_ResizeNearestNeighborOptions()->align_corners();

    auto pre_trans = nhwc_to_nchw(dt_float32, get_shape(input.shape()));
    pre_trans->name(get_tensor(op.outputs(), 0).name()->string_view());
    auto node = graph_.emplace<resize_image>(pre_trans->output().type(), mode, pre_trans->output().shape(), new_size, align_corners);
    node->name(get_tensor(op.outputs(), 0).name()->string_view());
    auto sur_trans = nchw_to_nhwc(dt_float32, node->output().shape());
    sur_trans->name(get_tensor(op.outputs(), 0).name()->string_view());

    node->input().connect(pre_trans->output());
    sur_trans->input().connect(node->output());

    auto input_conn = &pre_trans->input();
    auto output_conn = &sur_trans->output();

    if (input.type() != tflite::TensorType_FLOAT32)
    {
        std::vector<input_connector *> inputs_conn = { input_conn };
        std::vector<quant_param_t> input_dequant_params = {
            quant_param_t(to_vector(*input.quantization()->zero_point()), to_vector(*input.quantization()->scale()))
        };
        std::vector<output_connector *> outputs_conn = { output_conn };
        std::vector<quant_param_t> output_quant_params = {
            quant_param_t(to_vector(*output.quantization()->zero_point()), to_vector(*output.quantization()->scale()))
        };
        with_quantize(to_data_type(input.type()), inputs_conn, input_dequant_params, outputs_conn, output_quant_params);
        input_conn = inputs_conn[0];
        output_conn = outputs_conn[0];
    }

    link_input_tensor(input_conn, op.inputs()->Get(0));
    link_output_tensor(op.outputs()->Get(0), output_conn);
}
