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
#include <nncase/targets/target.h>

using namespace nncase;

target_options &target::options() {
    if (!options_)
        options_ = on_create_options();
    return *options_;
}

void target::configure_options([[maybe_unused]] const attribute_map &attrs) {}

void target::configure_passes_pre_schedule(
    [[maybe_unused]] ir::transforms::pass_manager &pmgr) {}

std::unique_ptr<target_options> target::on_create_options() {
    return std::make_unique<target_options>();
}
