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
#include <nncase/ir/op_utils.h>
#include <nncase/ir/ops/reduce_window2d.h>

using namespace nncase;
using namespace nncase::ir;

reduce_window2d::reduce_window2d(reduce_op_t reduce_op, shape_t input_shape, float init_value, int32_t filter_h, int32_t filter_w, padding padding_h, padding padding_w, int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w, value_range<float> fused_activation, bool ceil_mode, bool count_include_pad, std::vector<int32_t> padding_h_w_after, bool strict_inside_input)
    : reduce_op_(reduce_op), init_value_(init_value), filter_h_(filter_h), filter_w_(filter_w), padding_h_(padding_h), padding_w_(padding_w), stride_h_(stride_h), stride_w_(stride_w), dilation_h_(dilation_h), dilation_w_(dilation_w), fused_activation_(fused_activation), ceil_mode_(ceil_mode), count_include_pad_(count_include_pad), padding_h_w_after_(padding_h_w_after), strict_inside_input_(strict_inside_input)
{
    add_input("input", dt_float32, input_shape);
    auto output_size_h = get_windowed_output_size((int32_t)input_shape[2] + padding_h_.sum(), filter_h_, stride_h_, dilation_h_, false, ceil_mode);
    auto output_size_w = get_windowed_output_size((int32_t)input_shape[3] + padding_w_.sum(), filter_w_, stride_w_, dilation_w_, false, ceil_mode);

    if (strict_inside_input)
    {
        if ((output_size_h - 1) * stride_h >= (int32_t)input_shape[2] - padding_h_w_after[0])
            output_size_h -= 1;
        if ((output_size_w - 1) * stride_w >= (int32_t)input_shape[3] - padding_h_w_after[1])
            output_size_w -= 1;
    }

    add_output("output", dt_float32,
        shape_t {
            input_shape[0],
            input_shape[1],
            output_size_h,
            output_size_w });
}

bool reduce_window2d::properties_equal(node &other) const
{
    auto &r = static_cast<reduce_window2d &>(other);
    return reduce_op() == r.reduce_op() && init_value() == r.init_value() && filter_h() == r.filter_h()
        && filter_w() == r.filter_w() && padding_h() == r.padding_h() && padding_w() == padding_w()
        && stride_h() == r.stride_h() && stride_w() == r.stride_w() && dilation_h() == r.dilation_h()
        && dilation_w() == r.dilation_w() && fused_activation() == r.fused_activation()
        && ceil_mode() == r.ceil_mode() && count_include_pad() == r.count_include_pad()
        && padding_h_w_after() == r.padding_h_w_after() && strict_inside_input() == r.strict_inside_input();
}
