/* Copyright 2019-2021 Canaan Inc.
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
#include <nncase/ir/ops/binary.h>
#include <nncase/ir/ops/bitcast.h>
#include <nncase/ir/ops/clamp.h>
#include <nncase/ir/ops/concat.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/convert.h>
#include <nncase/ir/ops/dequantize.h>
#include <nncase/ir/ops/pad.h>
#include <nncase/ir/ops/resize_image.h>
#include <nncase/ir/ops/slice.h>
#include <nncase/ir/ops/transpose.h>
#include <nncase/ir/visitor.h>
#include <nncase/transforms/neutral/pre_process_setting.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;

datatype_t get_datatype(std::string name)
{
    if (name == "uint8")
        return datatype_t::dt_uint8;
    else if (name == "int8")
        return datatype_t::dt_int8;
    return datatype_t::dt_float32;
}

bool pre_process_transform::on_try_match(node &node, transform_context &context)
{
    if (auto in_node = node_cast<input_node>(node))
    {
        if (in_node->output().type() == dt_float32 && scales_[0] != 0.f)
        {
            context.outputs.emplace_back(&in_node->output());
            context.matched_nodes.emplace_back(in_node);
            return true;
        }
    }
    return false;
}

void pre_process_transform::process(transform_context &context)
{
    auto old_inputs = context.outputs[0]->connections();
    auto old_in = node_cast<input_node>(*context.matched_nodes[0]);
    shape_t new_shape, old_shape;
    if (real_layout_ == "NHWC")
    {
        new_shape = { size_t(old_in->output().shape()[0]), size_t(input_shape_[1]), size_t(input_shape_[2]), size_t(input_shape_[3]) };
        old_shape = { size_t(old_in->output().shape()[0]), size_t(old_in->output().shape()[3]), size_t(old_in->output().shape()[1]), size_t(old_in->output().shape()[2]) };
    }
    else
    { //TODO : fit onnx
        new_shape = { size_t(old_in->output().shape()[0]), size_t(input_shape_[1]), size_t(input_shape_[2]), size_t(input_shape_[3]) };
        old_shape = { size_t(old_in->output().shape()[0]), size_t(old_in->output().shape()[1]), size_t(old_in->output().shape()[2]), size_t(old_in->output().shape()[3]) };
    }
    auto new_input = context.graph.emplace<input_node>(get_datatype(input_type_), new_shape);
    new_input->name("new_input");

    output_connector *mid_ptr;

    mid_ptr = &new_input->output();
    if (real_layout_ == "NHWC")
    {
        auto transpose_pre = context.graph.emplace<transpose>(mid_ptr->type(), mid_ptr->shape(), axis_t { 0, 3, 1, 2 });
        transpose_pre->name("NHWC_2_NCWH");
        transpose_pre->input().connect(*mid_ptr);
        mid_ptr = &transpose_pre->output();
    }

    // BGR2RGB : input_layout ,image_format_
    std::cout << "BGR:" << std::endl;
    if (image_format_ == "BGR")
    {

        std::vector<shape_t> concat_shapes { 3, shape_t { (mid_ptr->shape()[0]), 1, (mid_ptr->shape()[2]), (mid_ptr->shape()[3]) } };
        auto concat_slice = context.graph.emplace<concat>(get_datatype(input_type_), concat_shapes, 1);
        concat_slice->name("BGR2RGB_concat_NCHW");
        for (int i = 0; i < 3; i++)
        {
            auto slice_input = context.graph.emplace<slice>(get_datatype(input_type_), mid_ptr->shape(),
                axis_t { 0, i, 0, 0 },
                axis_t { int(mid_ptr->shape()[0]), i + 1, int(mid_ptr->shape()[2]), int(mid_ptr->shape()[3]) });
            slice_input->name("BGR2RGB_slice_NCHW_" + std::to_string(i));
            slice_input->input().connect(*mid_ptr);
            concat_slice->input_at(2 - i).connect(slice_input->output());
        }

        mid_ptr = &concat_slice->output();
    }

    // letterbox :
    /**
         * input_layout:  HW have different axis 
         * input_type:  pad value different 
         * input_range:{min, max} caculate pad value //uint8 pad 114, float pad min+(max-min)*(114/255)
         **/
    std::cout << "letterbox:" << std::endl;
    // model input shape :old_in->output().shape();
    if (old_in->output().shape() != new_shape)
    {
        [[maybe_unused]] int min = input_range_[0], max = input_range_[1];
        size_t model_h = old_shape[2];
        size_t model_w = old_shape[3];
        std::array<int32_t, 2> resize_shape { (int32_t)model_h, (int32_t)model_w };

        auto H = mid_ptr->shape()[2];
        auto W = mid_ptr->shape()[3];

        // float ratio = std::min(float(H) / model_h, float(W) / model_w);
        float ratio = std::min(model_h / float(H), model_w / float(W));
        std::vector<padding> pad_size { 4, padding { 0, 0 } };
        auto pad_H = int(model_h / ratio - H);
        auto pad_W = int(model_w / ratio - W);

        pad_size[2] = { int(pad_H / 2), pad_H - int(pad_H / 2) };
        pad_size[3] = { int(pad_W / 2), pad_W - int(pad_W / 2) };

        scalar pad_value;
        if (input_type_ == "uint8")
        {
            pad_value = 114;
        }
        else if (input_type_ == "int8")
        {
            pad_value = 114 - 128;
        }
        else
        {
            pad_value = float(input_range_[0] + (input_range_[1] - input_range_[0]) * (114.0 / 255));
        }
        auto letter_box_pad = context.graph.emplace<pad>(get_datatype(input_type_), mid_ptr->shape(), pad_size, pad_constant, pad_value);
        // auto input_resize = context.graph.emplace<bitcast>(get_datatype(input_type_), letter_box_pad->output().shape(), old_in->output().shape());
        auto input_resize = context.graph.emplace<resize_image>(letter_box_pad->output().type(), image_resize_bilinear, letter_box_pad->output().shape(), resize_shape);
        letter_box_pad->name("letterbox_pad");
        input_resize->name("letterbox_resize");
        letter_box_pad->input().connect(*mid_ptr);
        input_resize->input().connect(letter_box_pad->output());
        mid_ptr = &input_resize->output();
    }

    //normalize : mean scale input_layout
    constant *mean, *scale;
    // if (input_layout_ == "NCHW")
    // {
    mean = context.graph.emplace<constant>(dt_float32, shape_t { 1, 3, 1, 1 }, means_);
    scale = context.graph.emplace<constant>(dt_float32, shape_t { 1, 3, 1, 1 }, scales_);
    // }
    // else
    // {
    //     mean = context.graph.emplace<constant>(dt_float32, shape_t { 1, 1, 1, 3 }, means_);
    //     scale = context.graph.emplace<constant>(dt_float32, shape_t { 1, 1, 1, 3 }, scales_);
    // }
    mean->name("normalize_mean");
    scale->name("normalize_scale");

    auto in_convert = context.graph.emplace<convert>(mid_ptr->type(), mid_ptr->shape(), dt_float32);
    in_convert->name("normalize_in_convert");
    in_convert->input().connect(*mid_ptr);

    auto normalize_sub = context.graph.emplace<binary>(binary_sub, in_convert->output().shape(), mean->output().shape(), value_range<float>::full());
    auto normalize_mul = context.graph.emplace<binary>(binary_mul, normalize_sub->output().shape(), scale->output().shape(), value_range<float>::full());
    normalize_sub->name(context.matched_nodes[0]->name() + "_norm_sub");
    normalize_mul->name(context.matched_nodes[0]->name() + "_norm_mul");
    normalize_sub->input_a().connect(in_convert->output());
    normalize_sub->input_b().connect(mean->output());
    normalize_mul->input_a().connect(normalize_sub->output());
    normalize_mul->input_b().connect(scale->output());

    auto out_convert = context.graph.emplace<convert>(normalize_mul->output().type(), normalize_mul->output().shape(), mid_ptr->type());
    out_convert->name("normalize_out_convert");
    out_convert->input().connect(normalize_mul->output());

    // add stop condition
    scales_[0] = 0.f;

    mid_ptr = &out_convert->output();

    // dequantize
    if (mid_ptr->type() != dt_float32)
    {
        size_t bits = 0;
        if (quant_type_ == "uint8")
        {
            bits = 8;
        }
        else
        {
            bits = 7;
        }
        value_range<float> range = { 0, 1 };
        auto Q_max = bits == 7 ? 127 : 255;
        auto Q_min = bits == 7 ? -128 : 0;
        auto scale = (range.max - range.min) / (Q_max - Q_min);
        auto bias = std::round((range.max * Q_min - range.min * Q_max) / (range.max - range.min));
        quant_param_t deq_params { static_cast<int32_t>(bias), scale };
        auto deq_input = context.graph.emplace<dequantize>(mid_ptr->type(), mid_ptr->shape(), dt_float32, deq_params);
        deq_input->name("dequantize_input");
        deq_input->input().connect(*mid_ptr);
        mid_ptr = &deq_input->output();
    }
    if (real_layout_ == "NHWC")
    {
        auto transpose_post = context.graph.emplace<transpose>(mid_ptr->type(), mid_ptr->shape(), axis_t { 0, 2, 3, 1 });
        transpose_post->name("NCHW_2_NHWC");
        transpose_post->input().connect(*mid_ptr);
        mid_ptr = &transpose_post->output();
    }

    for (auto &in : dup(old_inputs))
        in->connect(*mid_ptr);
}
