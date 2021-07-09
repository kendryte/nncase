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
#include <nncase/ir/ops/bitcast.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace caffe;

DEFINE_CAFFE_LOWER(Reshape)
{
    // check if there are bn/scale/relu above
    std::string input_name = get_real_input_names(op)[0];

    auto &input = *output_tensors_.at(input_name);
    auto &param = op.reshape_param();

    // normalize new shape
    shape_t normalized_new_shape(get_axis(param.shape()).size());
    size_t shape_size = 1;
    std::optional<size_t> non_det_id;
    for (size_t i = 0; i < get_axis(param.shape()).size(); i++)
    {
        auto v = get_axis(param.shape())[i];
        if (v == -1)
        {
            if (non_det_id)
                throw std::runtime_error("Reshape can only have 1 non-determined dimension at most");
            non_det_id = i;
        }
        else if(v == 0)
        {
            shape_size *= input.shape()[i];
            normalized_new_shape[i] = (size_t)input.shape()[i];
        }
        else
        {
            shape_size *= v;
            normalized_new_shape[i] = (size_t)get_axis(param.shape())[i];
        }
    }
    if (non_det_id)
        normalized_new_shape[*non_det_id] = xt::compute_size(input.shape()) / shape_size;

    auto rp = graph_.emplace<bitcast>(dt_float32, input.shape(), normalized_new_shape);
    rp->name(op.name() + "/bitcast");

    input_tensors_.emplace(&rp->input(), input_name);
    output_tensors_.emplace(op.top(0), &rp->output());
}
