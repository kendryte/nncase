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
#include <nncase/ir/debug.h>
#include <nncase/ir/visitor.h>
#include <nncase/transforms/pass.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;

void function_pass::run(const function &func, const run_pass_options &options) {
    run_core(func, options);
    if (options.dump_dir) {
        auto dump_path = *options.dump_dir / "passes" / name();
        std::filesystem::create_directories(dump_path);
        ir::dump_function(func, dump_path);
    }
}

function_pass &pass_manager::emplace(std::unique_ptr<function_pass> pass) {
    return *passes_.emplace_back(std::move(pass));
}

void pass_manager::run() {
    for (auto &pass : passes_)
        pass->run(func_, options_);
}
