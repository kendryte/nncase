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
#include <algorithm>
#include <filesystem>
#include <nncase/ir/visitor.h>
#include <nncase/transforms/egraph.h>
#include <nncase/transforms/egraph_pass.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;

void egraph_pass::run_core(const function &func,
                           const run_pass_options &options) {
    egraph graph;
    graph.add(func);
}
