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
#include "../caffe_importer.h"
#include <nncase/ir/ops/slice.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace caffe;

DEFINE_CAFFE_LOWER(Slice)
{
    // check if there are bn/scale/relu above
    std::string input_name = get_real_input_names(op)[0];

    auto &input = *output_tensors_.at(input_name);
    auto &param = op.slice_param();
    auto in_shape = input.shape();
    axis_t begin_templ(in_shape.size());
    axis_t end_templ(begin_templ.size());
    for (size_t i = 0; i < end_templ.size(); i++)
        end_templ[i] = in_shape[i];
    axis_t strides(begin_templ.size());
    for (size_t i = 0; i < strides.size(); i++)
        strides[i] = 1;

    int32_t axis_beg = 0;
    for (int i = 0; i < op.top_size(); i++)
    {
        auto begin = begin_templ;
        auto end = end_templ;
        begin[param.axis()] = axis_beg;
        if (i != op.top_size() - 1)
            axis_beg = end[param.axis()] = axis_beg + param.slice_point(i);

        auto sl = graph_.emplace<slice>(dt_float32, in_shape, begin, end, strides, 0, 0, 0, 0);
        sl->name(op.name() + "/slice");
        input_tensors_.emplace(&sl->input(), input_name);
        output_tensors_.emplace(op.top(i), &sl->output());
    }
}
