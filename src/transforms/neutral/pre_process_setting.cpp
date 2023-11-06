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

void pre_process_transform::run_core(graph &graph, [[maybe_unused]] nncase::target &target, [[maybe_unused]] const run_pass_options &options)
{
    for (auto in_node : dup(graph.inputs()))
    {
        if (in_node->output().shape().size() == 4)
        {
            auto old_inputs = dup(in_node->output().connections());
            shape_t new_shape = { size_t(in_node->output().shape()[0]), size_t(input_shape_[1]), size_t(input_shape_[2]), size_t(input_shape_[3]) };
            auto new_input = graph.emplace<input_node>(get_datatype(input_type_), new_shape);
            new_input->name("new_input");

            output_connector *mid_ptr;

            mid_ptr = &new_input->output();

            if (input_layout_ == "NHWC")
            {
                auto transpose_pre = graph.emplace<transpose>(mid_ptr->type(), mid_ptr->shape(), axis_t { 0, 3, 1, 2 });
                transpose_pre->name("NHWC_2_NCWH");
                transpose_pre->input().connect(*mid_ptr);
                mid_ptr = &transpose_pre->output();
            }

            // swapRB : input_layout ,swapRB_
            if (swapRB_ == true && mid_ptr->shape()[1] == 3)
            {
                std::cout << " |Exchange image channel:" << std::endl;
                std::vector<shape_t> concat_shapes { 3, shape_t { (mid_ptr->shape()[0]), 1, (mid_ptr->shape()[2]), (mid_ptr->shape()[3]) } };
                auto concat_slice = graph.emplace<concat>(mid_ptr->type(), concat_shapes, 1);
                concat_slice->name("swapRB_concat_NCHW");
                for (int i = 0; i < 3; i++)
                {
                    auto slice_input = graph.emplace<slice>(mid_ptr->type(), mid_ptr->shape(),
                        axis_t { 0, i, 0, 0 },
                        axis_t { int(mid_ptr->shape()[0]), i + 1, int(mid_ptr->shape()[2]), int(mid_ptr->shape()[3]) });
                    slice_input->name("swapRB_slice_NCHW_" + std::to_string(i));
                    slice_input->input().connect(*mid_ptr);
                    concat_slice->input_at(2 - i).connect(slice_input->output());
                }

                mid_ptr = &concat_slice->output();
            }

            //dequantize: input_range_
            if (mid_ptr->type() != dt_float32)
            {
                std::cout << " |Dequantize:" << std::endl;
                value_range<float> range = { input_range_[0], input_range_[1] };

                auto Q_max = 255;
                auto Q_min = 0;
                auto scale = (range.max - range.min) / (Q_max - Q_min);
                auto bias = std::round((range.max * Q_min - range.min * Q_max) / (range.max - range.min));
                quant_param_t deq_params { static_cast<int32_t>(bias), scale };
                auto deq_input = graph.emplace<dequantize>(mid_ptr->type(), mid_ptr->shape(), dt_float32, deq_params);
                deq_input->name("dequantize_input");
                deq_input->input().connect(*mid_ptr);
                mid_ptr = &deq_input->output();
            }

            // letterbox :
            /**
             * input_layout:  HW have different axis
             * input_type:  pad value different
             *  //input_range:{min, max} caculate pad value //uint8 pad 114, float pad min+(max-min)*(114/255)
             **/

            if (in_node->output().shape() != new_shape)
            {
                if ((real_inlayout_ == "NHWC" && in_node->output().shape() != shape_t { new_shape[0], new_shape[2], new_shape[3], new_shape[1] })
                    || (real_inlayout_ == "NCHW" && in_node->output().shape() != shape_t { new_shape[0], new_shape[3], new_shape[1], new_shape[2] }))
                {
                    std::cout << " |Letterbox:" << std::endl;
                    size_t model_h;
                    size_t model_w;
                    if (real_inlayout_ == "NHWC")
                    {
                        model_h = in_node->output().shape()[1];
                        model_w = in_node->output().shape()[2];
                    }
                    else
                    {
                        model_h = in_node->output().shape()[2];
                        model_w = in_node->output().shape()[3];
                    }

                    auto H = mid_ptr->shape()[2];
                    auto W = mid_ptr->shape()[3];

                    float ratio = std::min(model_h / float(H), model_w / float(W));
                    std::vector<padding> pad_size { 4, padding { 0, 0 } };
                    auto resize_H = std::round(H * ratio);
                    auto resize_W = std::round(W * ratio);

                    int pad_H = model_h - resize_H;
                    int pad_W = model_w - resize_W;
                    std::array<int32_t, 2> resize_shape { (int32_t)resize_H, (int32_t)resize_W };

                    pad_size[2] = { int(std::round(pad_H / 2 - 0.1)), pad_H - int(std::round(pad_H / 2 - 0.1)) };
                    pad_size[3] = { int(std::round(pad_W / 2 - 0.1)), pad_W - int(std::round(pad_W / 2 - 0.1)) };

                    scalar pad_value = letterbox_value_;
                    auto input_resize = graph.emplace<resize_image>(mid_ptr->type(), image_resize_bilinear, mid_ptr->shape(), resize_shape, false, true);
                    auto letter_box_pad = graph.emplace<pad>(input_resize->output().type(), input_resize->output().shape(), pad_size, pad_constant, pad_value);
                    input_resize->name("letterbox_resize");
                    letter_box_pad->name("letterbox_pad");
                    input_resize->input().connect(*mid_ptr);
                    letter_box_pad->input().connect(input_resize->output());
                    mid_ptr = &letter_box_pad->output();
                }
            }

            //normalize : mean scale input_layout
            if (std_[0] != 0)
            {
                std::cout << " |Normalize:" << std::endl;
                constant *mean, *scale;
                if (mid_ptr->shape()[1] != 3)
                {
                    auto single_mean = mean_[0];
                    mean = graph.emplace<constant>(single_mean);
                    auto single_scale = std_[0];
                    scale = graph.emplace<constant>(single_scale);
                }
                else
                {
                    mean = graph.emplace<constant>(dt_float32, shape_t { 1, 3, 1, 1 }, mean_);
                    scale = graph.emplace<constant>(dt_float32, shape_t { 1, 3, 1, 1 }, std_);
                }
                mean->name("normalize_mean");
                scale->name("normalize_scale");

                auto in_convert = graph.emplace<convert>(mid_ptr->type(), mid_ptr->shape(), dt_float32);
                in_convert->name("normalize_in_convert");
                in_convert->input().connect(*mid_ptr);

                auto normalize_sub = graph.emplace<binary>(binary_sub, in_convert->output().type(), in_convert->output().shape(), mean->output().shape(), value_range<float>::full());
                auto normalize_mul = graph.emplace<binary>(binary_div, normalize_sub->output().type(), normalize_sub->output().shape(), scale->output().shape(), value_range<float>::full());
                normalize_sub->name("input_norm_sub");
                normalize_mul->name("input_norm_div");
                normalize_sub->input_a().connect(in_convert->output());
                normalize_sub->input_b().connect(mean->output());
                normalize_mul->input_a().connect(normalize_sub->output());
                normalize_mul->input_b().connect(scale->output());

                auto out_convert = graph.emplace<convert>(normalize_mul->output().type(), normalize_mul->output().shape(), mid_ptr->type());
                out_convert->name("normalize_out_convert");
                out_convert->input().connect(normalize_mul->output());
                mid_ptr = &out_convert->output();
            }

            if (real_inlayout_ == "NHWC")
            {
                auto transpose_post = graph.emplace<transpose>(mid_ptr->type(), mid_ptr->shape(), axis_t { 0, 2, 3, 1 });
                transpose_post->name("NCHW_2_NHWC");
                transpose_post->input().connect(*mid_ptr);
                mid_ptr = &transpose_post->output();
            }

            for (auto &in : old_inputs)
                in->connect(*mid_ptr);
        }
        graph.dce();
    }
}
