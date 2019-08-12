/* Copyright 2019 Canaan Inc.
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
#include "../paddle_importer.h"
#include <ir/ops/conv2d.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace paddle::framework::proto;

DEFINE_PADDLE_LOWER(conv2d)
{
    auto &input = find_var(op.inputs(), "Input");
    auto &weights = find_var(op.inputs(), "Filter");
    auto &output = find_var(op.outputs(), "Output");
    auto w_shape = get_var_shape(weights);

    auto &paddings = find_attr(op.attrs(), "paddings").ints();
    xt::svector<padding> new_paddings;
    for (size_t i = 0; i < paddings.size(); i++)
        new_paddings.push_back(padding { paddings[i], paddings[i] });
    auto &strides = find_attr(op.attrs(), "strides").ints();
    auto &dilations = find_attr(op.attrs(), "dilations").ints();
    auto groups = find_attr(op.attrs(), "groups").i();
    auto weights_tensor = load_tensor<float, 4>(weights);
    xt::xtensor<float, 1> bias(std::array<size_t, 1> { w_shape[0] }, 0.f);

    auto node = graph_.emplace<conv2d>(get_var_shape(input), weights_tensor, bias, groups, new_paddings[0], new_paddings[1],
        strides[0], strides[1], dilations[0], dilations[1], value_range<float>::full());

    input_tensors_.emplace(&node->input(), input.name());
    output_tensors_.emplace(output.name(), &node->output());
}
