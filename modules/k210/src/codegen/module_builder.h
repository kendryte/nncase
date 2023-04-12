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
#include <nncase/codegen/k210/module_builder.h>
#include <nncase/ir/ops/copy.h>
#include <nncase/ir/ops/k210/kpu_conv2d.h>
#include <nncase/ir/ops/k210/kpu_data_exchange.h>
#include <nncase/ir/placeholders.h>
#include <nncase/schedule/scheduler.h>

namespace nncase::codegen::k210 {
class k210_module_builder : public module_builder {
  public:
    k210_module_builder(std::string_view module_name,
                        const module_builder_params &params);

    module_type_t module_type() const noexcept override;
    uint32_t module_version() const noexcept override;

  protected:
    section_writer &text_writer();

    void begin_emit_function(
        const schedule::function_schedule_result &function) override;
    void end_emit_function(
        const schedule::function_schedule_result &function) override;
    void emit(ir::node &node) override;

  private:
#define DEFINE_OP(op_) void emit(ir::op_ &op);
#include "ops.def"
#undef DEFINE_OP
};
} // namespace nncase::codegen::k210
