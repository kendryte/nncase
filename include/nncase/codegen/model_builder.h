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
#include "module_builder.h"

namespace nncase::codegen
{
struct build_model_result
{
    size_t model_size;
};

class NNCASE_API model_builder
{
public:
    model_builder(target &target, const schedule::schedule_result &sched);
    model_builder(model_builder &) = delete;
    model_builder(model_builder &&) = delete;

    void config_dump(const std::filesystem::path &dump_dir, bool dump_asm);
    build_model_result build(std::ostream &output);

    size_t max_usage(memory_location_t location) const;

private:
    target &target_;
    const schedule::schedule_result &sched_;
    std::filesystem::path dump_dir_;
    bool dump_asm_;
};
}
