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
#include <nncase/ir/op_utils.h>
#include <nncase/ir/ops/call.h>

using namespace nncase;
using namespace nncase::ir;

call::call(graph &target)
    : target_(target)
{
    size_t i = 0;
    for (auto &in : target_.inputs())
        add_input(in->name(), in->output().type(), in->output().shape());

    i = 0;
    for (auto &out : target_.outputs())
        add_output(out->name(), out->input().type(), out->input().shape());
}

bool call::properties_equal(node &other) const
{
    auto &r = static_cast<call &>(other);
    return &target() == &r.target();
}
