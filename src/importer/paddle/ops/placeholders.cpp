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
#include <ir/placeholders.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace paddle::framework::proto;

DEFINE_PADDLE_LOWER(feed)
{
    auto &output = find_var(op.outputs(), "Out");

    auto node = graph_.emplace<input_node>(to_data_type(get_lod_tensor_type(output)), get_var_shape(output));
    node->name(output.name());

    output_tensors_.emplace(output.name(), &node->output());
}
