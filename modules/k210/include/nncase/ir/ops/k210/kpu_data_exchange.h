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
#include <nncase/ir/node.h>
#include <nncase/ir/ops/k210/opcode.h>
#include <nncase/runtime/k210/runtime_types.h>
#include <xtensor/xtensor.hpp>

namespace nncase::ir::k210 {
class NNCASE_MODULES_K210_API kpu_upload : public node {
  public:
    DEFINE_NODE_OPCODE(op_k210_kpu_upload);

    input_connector &input() { return input_at(0); }
    output_connector &output() { return output_at(0); }

    kpu_upload(shape_t input_shape);

  protected:
    bool properties_equal([[maybe_unused]] node &other) const override {
        return true;
    }
};

class NNCASE_MODULES_K210_API kpu_download : public node {
  public:
    DEFINE_NODE_OPCODE(op_k210_kpu_download);

    input_connector &input() { return input_at(0); }
    output_connector &output() { return output_at(0); }

    kpu_download(shape_t input_shape);

  protected:
    bool properties_equal([[maybe_unused]] node &other) const override {
        return true;
    }
};
} // namespace nncase::ir::k210
