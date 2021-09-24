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
#include <memory>
#include <nncase/attribute_map.h>
#include <nncase/runtime/datatypes.h>

namespace nncase::codegen {
struct module_builder_params;
class module_builder;
} // namespace nncase::codegen

namespace nncase::ir::transforms {
class pass_manager;
}

namespace nncase {
struct NNCASE_API target_options {
    virtual ~target_options() = default;

    std::string input_type;
    std::string output_type;
    std::string inference_type;
    std::string weights_type;
};

class NNCASE_API target {
  public:
    virtual ~target() = default;

    target_options &options();
    virtual void configure_options(const attribute_map &attrs);

    virtual void
    configure_passes_pre_schedule(ir::transforms::pass_manager &pmgr);

  protected:
    virtual std::unique_ptr<target_options> on_create_options();

  private:
    std::unique_ptr<target_options> options_;
};
} // namespace nncase
