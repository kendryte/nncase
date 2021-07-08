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
#include <functional>
#include <nncase/ir/ops/pad.h>
#include <nncase/ir/ops/slice.h>
#include <nncase/ir/ops/constant.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace caffe;

DEFINE_CAFFE_LOWER(Reverse)
{
    // check if there are bn/scale/relu above
    std::string input_name = get_real_input_names(op)[0];

    auto &input = *output_tensors_.at(input_name);
    auto &param = op.reverse_param();

    auto axis = param.axis();
    xt::svector<padding> new_paddings;
    for (size_t i = 0; i < input.shape().size(); i++)
    {
        if ((int32_t)i == axis)
            new_paddings.push_back(padding { (int32_t)input.shape()[i], 0 });
        else
            new_paddings.push_back(padding { 0, 0 });
    }
    auto p = graph_.emplace<pad>(input.type(), input.shape(), new_paddings, pad_symmetric, 0.f);
    p->name(op.name() + "/pad");

    axis_t begin(p->output().shape().size());
    for (size_t i = 0; i < begin.size(); i++)
        begin[i] = 0;
    axis_t end(begin.size());
    for (size_t i = 0; i < end.size(); i++)
    {
        if ((int32_t)i != axis)
            end[i] = p->output().shape()[i];
        else
            end[i] = p->output().shape()[i] / 2;
    }
    axis_t strides(begin.size());
    for (size_t i = 0; i < strides.size(); i++)
        strides[i] = 1;

    auto sl = graph_.emplace<slice>(input.type(), p->output().shape(), begin, end, strides, 0, 0, 0, 0);
    sl->name(op.name() + "/slice");

    input_tensors_.emplace(&p->input(), input_name);
    sl->input().connect(p->output());
    output_tensors_.emplace(op.top(0), &sl->output());
}
