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
#include "schedule_types.h"
#include <filesystem>
#include <span>

namespace nncase
{
class target;

namespace schedule
{
    class NNCASE_API scheduler
    {
    public:
        scheduler(target &target, ir::graph &main_graph, std::span<ir::output_node *> outputs)
            : target_(target), main_graph_(main_graph), outputs_(outputs) { }

        schedule_result schedule(bool skip_buffer_alias = false);
        void config_dump(std::filesystem::path dump_dir);

    private:
        void dump_schedule(const schedule_context &context);

    private:
        target &target_;
        ir::graph &main_graph_;
        std::span<ir::output_node *> outputs_;
        std::filesystem::path dump_dir_;
    };
}
}
