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
#include "../caffe_importer.h"
#include <nncase/ir/ops/binary.h>
#include <nncase/ir/ops/unary.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace caffe;

DEFINE_CAFFE_LOWER(Eltwise)
{
    // check if there are bn/scale/relu above
    std::vector<std::string> input_names;
    std::vector<ir::output_connector *> inputs;
    for (size_t i = 0; i < get_real_input_names(op).size(); i++)
    {
        input_names.push_back(get_real_input_names(op)[i]);
        inputs.push_back(output_tensors_.at(input_names[i]));
    }
    auto &param = op.eltwise_param();

    if (param.operation() == EltwiseParameter_EltwiseOp_PROD)
    {
        std::vector<ir::node *> muls;
        for (size_t i = 0; i < inputs.size() - 1; i++)
        {
            auto mul = graph_.emplace<binary>(binary_mul, inputs[0]->shape(), inputs[0]->shape(), value_range<float>::full());
            mul->name(op.name() + "/mul");
            if (i == 0)
            {
                input_tensors_.emplace(&mul->input_a(), input_names[0]);
                input_tensors_.emplace(&mul->input_b(), input_names[1]);
            }
            else
            {
                input_tensors_.emplace(&mul->input_b(), input_names[i + 1]);
                mul->input_a().connect(muls.back()->output_at(0));
            }
            if (i == inputs.size() - 2)
                output_tensors_.emplace(op.top(0), &mul->output());
            muls.push_back(mul);
        }
    }
    else if (param.operation() == EltwiseParameter_EltwiseOp_SUM)
    {
        std::vector<ir::node *> adds;
        for (size_t i = 0; i < inputs.size() - 1; i++)
        {
            auto add = graph_.emplace<binary>(binary_add, inputs[0]->shape(), inputs[0]->shape(), value_range<float>::full());
            add->name(op.name() + "/add");
            if (i == 0)
            {
                if (param.coeff().size() != 0 && param.coeff().size() != inputs.size())
                    throw std::runtime_error("coeff size must align with input shape size");

                if (param.coeff().size() == 0 || param.coeff()[0] == 1)
                    input_tensors_.emplace(&add->input_a(), input_names[0]);
                else
                {
                    auto neg = graph_.emplace<unary>(unary_neg, inputs[0]->shape());
                    neg->name(op.name() + "/neg");
                    input_tensors_.emplace(&neg->input(), input_names[0]);
                    add->input_a().connect(neg->output());
                }
                if (param.coeff().size() == 0 || param.coeff()[1] == 1)
                    input_tensors_.emplace(&add->input_b(), input_names[1]);
                else
                {
                    auto neg = graph_.emplace<unary>(unary_neg, inputs[0]->shape());
                    neg->name(op.name() + "/neg");
                    input_tensors_.emplace(&neg->input(), input_names[1]);
                    add->input_b().connect(neg->output());
                }
            }
            else
            {
                if (param.coeff().size() == 0 || param.coeff()[i + 1] == 1)
                {
                    input_tensors_.emplace(&add->input_b(), input_names[i + 1]);
                }
                else
                {
                    auto neg = graph_.emplace<unary>(unary_neg, inputs[0]->shape());
                    neg->name(op.name() + "/neg");
                    input_tensors_.emplace(&neg->input(), input_names[i + 1]);
                    add->input_b().connect(neg->output());
                }
                add->input_a().connect(adds.back()->output_at(0));
            }
            if (i == inputs.size() - 2)
                output_tensors_.emplace(op.top(0), &add->output());
            adds.push_back(add);
        }
    }
    else if (param.operation() == EltwiseParameter_EltwiseOp_MAX)
    {
        std::vector<ir::node *> maxes;
        for (size_t i = 0; i < inputs.size() - 1; i++)
        {
            auto max = graph_.emplace<binary>(binary_max, inputs[0]->shape(), inputs[0]->shape(), value_range<float>::full());
            max->name(op.name() + "/max");
            if (i == 0)
            {
                input_tensors_.emplace(&max->input_a(), input_names[0]);
                input_tensors_.emplace(&max->input_b(), input_names[1]);
            }
            else
            {
                input_tensors_.emplace(&max->input_b(), input_names[i + 1]);
                max->input_a().connect(maxes.back()->output_at(0));
            }
            if (i == inputs.size() - 2)
                output_tensors_.emplace(op.top(0), &max->output());
            maxes.push_back(max);
        }
    }
    else
        throw std::runtime_error("invalid elementwise op");
}
