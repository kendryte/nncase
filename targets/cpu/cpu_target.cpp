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
#include "cpu_target.h"
#include <nncase/plugin_loader.h>
#include <nncase/transforms/neutral/fold_constant.h>
#include <nncase/transforms/neutral/lstm_transform.h>
#include <nncase/transforms/pass.h>

#if defined(_MSC_VER)
#define CPU_TARGET_API __declspec(dllexport)
#else
#define CPU_TARGET_API __attribute__((visibility("default")))
#endif

using namespace nncase;
using namespace nncase::targets;
using namespace nncase::runtime;
using namespace nncase::ir::transforms;

extern "C"
{
    CPU_TARGET_API target *create_target()
    {
        return new cpu_target();
    }
}

void cpu_target::register_target_dependent_passes([[maybe_unused]] const module_type_t &type, ir::transforms::pass_manager &pass_mgr, [[maybe_unused]] bool use_ptq)
{
    // lstm_transform
    {
        transform_pass p("lstm_transform");
        p.emplace<fold_constant_transform>();
        p.emplace<lstm_transform>();
        pass_mgr.add_pass(std::move(p));
    }
}