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
#pragma once
#include <nncase/codegen/module_builder.h>
#include <nncase/runtime/k210/runtime_module.h>

namespace nncase::codegen {
NNCASE_MODULES_K210_API std::unique_ptr<module_builder>
create_k210_module_builder(std::string_view module_name,
                           const module_builder_params &params);
}
