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
#include "../transform.h"

namespace nncase::ir::transforms
{
class NNCASE_API add_input_dequantize_transform : public transform
{
public:
    add_input_dequantize_transform(datatype_t dt) noexcept
        : input_type_(dt) { }
    void process(transform_context &context) override;

protected:
    bool skip_self_contained_check() const noexcept override { return true; }
    bool on_try_match(ir::node &node, transform_context &context) override;

private:
    datatype_t input_type_;
};

class NNCASE_API add_output_quantize_transform : public transform
{
public:
    add_output_quantize_transform(datatype_t dt, quant_param_t &output_quant_param) noexcept
        : output_type_(dt), output_quant_param_(output_quant_param) { }
    void process(transform_context &context) override;

protected:
    bool skip_self_contained_check() const noexcept override { return true; }
    bool on_try_match(ir::node &node, transform_context &context) override;

private:
    datatype_t output_type_;
    quant_param_t &output_quant_param_;
};
}
