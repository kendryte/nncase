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
#include "../tflite_importer.h"
#include <flatbuffers/flexbuffers.h>
#include <nncase/ir/ops/random_uniform.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

DEFINE_TFLITE_LOWER(CUSTOM)
{
    auto opcode = model_->operator_codes()->Get(op.opcode_index());
    auto custom_code = opcode->custom_code()->str();
    std::cout << "custom_code = " << custom_code << std::endl;
    if (custom_code == "FlexRandomUniform")
    {
        // auto custom_options = op.custom_options();
        // auto r = flexbuffers::GetRoot(custom_options->data(), custom_options->size());
        // std::cout << "Reference type: " << r.GetType() << std::endl;
        // auto v = flexbuffers::GetRoot(custom_options->data(), custom_options->size()).AsVector();
        // std::cout << "v.size = " << v.size() << std::endl;
        // for (size_t i = 0; i < v.size(); i++)
        // {
        //     std::cout << "i = " << i << ": Reference type: " << v[i].GetType() << std::endl;
        //     if (v[i].IsString())
        //         std::cout << v[i].AsString().str() << std::endl;
        // }
        auto &output = get_tensor(op.outputs(), 0);
        auto node = graph_.emplace<random_uniform>(to_data_type(output.type()), get_shape(output.shape()), 0.f, 1.f, time(nullptr));
        node->name(output.name()->string_view());
        link_output_tensor(op.outputs()->Get(0), &node->output());
    }
    else
    {
        throw std::runtime_error(std::string("Unsupported tflite CUSTOM code: ") + custom_code);
    }
}
