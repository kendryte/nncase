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
#include "tflite_importer.h"
//#include <nncase/importer/util.h>
#include <nncase/ir/constant.h>
#include <nncase/ir/math/functional.h>
#include <nncase/ir/tuple.h>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/transform.hpp>
//#include <nncase/ir/ops/convert.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace flatbuffers;

tflite_importer::tflite_importer(std::span<const uint8_t> model)
    : model_(tflite::GetModel(model.data())),
      subgraph_(model_->subgraphs()->Get(0)) {
    flatbuffers::Verifier verifier(model.data(), model.size());
    if (!tflite::VerifyModelBuffer(verifier))
        throw std::runtime_error("Invalid tflite model");
}

module_t
tflite_importer::import([[maybe_unused]] const import_options &options) {
    // 1. Create inputs
    std::vector<var> created_inputs;
    for (auto in : *subgraph_->inputs()) {
        auto tensor = subgraph_->tensors()->Get(in);
        var input(tensor->name()->str(),
                  get_ir_type(tensor->shape(), tensor->type()));
        created_inputs.emplace_back(input);
        output_tensors_.emplace(in, input);
    }

    // 2. Convert ops
    auto &operators = *subgraph_->operators();
    for (auto op : operators)
        convert_op(*op);

    // 3. Create outputs
    std::vector<expr> output_exprs;
    output_exprs.reserve(subgraph_->outputs()->size());
    for (auto out : *subgraph_->outputs())
        output_exprs.emplace_back(output_tensors_.at(out));
    tuple output(std::move(output_exprs));

    // 4. Create function
    function main_func("main", created_inputs, output);
    module_->functions().emplace_back(main_func);
    module_->entry(main_func);
    return module_;
}

ir::shape_t
tflite_importer::get_ir_shape(const flatbuffers::Vector<int32_t> *shape) {
    if (shape) {
        return *shape | ranges::views::transform([](int32_t dim) -> dim_t {
            return dim;
        }) | ranges::to<std::vector>();
    } else {
        return unranked_shape;
    }
}

ir::type tflite_importer::get_ir_type(const flatbuffers::Vector<int32_t> *shape,
                                      tflite::TensorType tflite_type) {
    return tensor_type(to_datatype(tflite_type), get_ir_shape(shape));
}

datatype_t tflite_importer::to_datatype(tflite::TensorType tflite_type) {
    switch (tflite_type) {
    case tflite::TensorType_FLOAT32:
        return dt_float32;
    case tflite::TensorType_FLOAT16:
        return dt_float16;
    case tflite::TensorType_INT32:
        return dt_int32;
    case tflite::TensorType_UINT8:
        return dt_uint8;
    case tflite::TensorType_INT64:
        return dt_int64;
    case tflite::TensorType_STRING:
        return dt_string;
    case tflite::TensorType_BOOL:
        return dt_bool;
    case tflite::TensorType_INT16:
        return dt_int16;
    // case tflite::TensorType_COMPLEX64 = 8,
    case tflite::TensorType_INT8:
        return dt_int8;
    case tflite::TensorType_FLOAT64:
        return dt_float64;
    // case tflite::TensorType_COMPLEX128 = 11,
    case tflite::TensorType_UINT64:
        return dt_uint64;
    // case tflite::TensorType_RESOURCE = 13,
    // case tflite::TensorType_VARIANT:
    case tflite::TensorType_UINT32:
        return dt_uint32;
    default:
        throw std::runtime_error(
            "Unsupported tensor type: " +
            std::string(tflite::EnumNameTensorType(tflite_type)));
    }
}

void tflite_importer::convert_op(const tflite::Operator &op) {
    auto opcode = model_->operator_codes()->Get(op.opcode_index());
    // Compatible with older version model
    auto builtin_code = static_cast<tflite::BuiltinOperator>(
        std::max(static_cast<int32_t>(opcode->deprecated_builtin_code()),
                 static_cast<int32_t>(opcode->builtin_code())));
#define DEFINE_OPCODE(opcode)                                                  \
    if (builtin_code == tflite::BuiltinOperator_##opcode)                      \
        return convert_op_##opcode(op);
#include "opcode.def"
#undef DEFINE_OPCODE

    throw std::runtime_error(std::string("Unsupported tflite opcode: ") +
                             tflite::EnumNameBuiltinOperator(builtin_code));
}

ir::expr
tflite_importer::get_tensor_expr(const flatbuffers::Vector<int32_t> *ids,
                                 int32_t offset) {
    auto id = ids->Get(offset);
    auto it = output_tensors_.find(id);
    if (it != output_tensors_.end()) {
        return it->second;
    } else { // Maybe constant
        auto &tensor = *subgraph_->tensors()->Get(id);
        auto &buffer = *model_->buffers()->Get(tensor.buffer());
        auto data = buffer.data();

        if (data) {
            auto ir_type = get_ir_type(tensor.shape(), tensor.type());
            constant con(ir_type,
                         std::span(data->data(), data->data() + data->size()));
            return output_tensors_.emplace(id, con).first->second;
        } else {
            throw std::runtime_error("Unable to load tflite tensor: " +
                                     tensor.name()->str());
        }
    }
}

ir::expr tflite_importer::get_input_expr(const tflite::Operator &op,
                                         int32_t offset) {
    return get_tensor_expr(op.inputs(), offset);
}

void tflite_importer::set_tensor_expr(const flatbuffers::Vector<int32_t> *ids,
                                      int32_t offset, ir::expr ex) {
    auto id = ids->Get(offset);
    output_tensors_[id] = ex;
}

void tflite_importer::set_output_expr(const tflite::Operator &op,
                                      int32_t offset, ir::expr ex) {
    set_tensor_expr(op.outputs(), offset, ex);
}

ir::call tflite_importer::activate(ir::expr input,
                                   tflite::ActivationFunctionType activation) {
    auto range = to_float_clamp_range(activation);
    return F::clamp(input, range.min, range.max);
}
//
// quant_param_t tflite_importer::to_quant_param(const
// tflite::QuantizationParameters *param)
//{
//    // TODO: consider of by axis quant
//    return { (int32_t)param->zero_point()->Get(0), param->scale()->Get(0) };
//}
//
// void tflite_importer::add_convert(ir::input_connector &next_input, const
// tflite::Tensor &tensor, int32_t tf_id, datatype_t to_type)
//{
//    auto ct = nncase::importer::add_prev_node<ir::convert>(graph_, next_input,
//    to_data_type(tensor.type()), get_shape(tensor.shape()), to_type);
//    link_input_tensor(&ct->input(), tf_id);
//}
//
// void tflite_importer::input_convert_to_type(ir::input_connector &next_input,
// const tflite::Tensor &tensor, int32_t tf_id, datatype_t to_type)
//{
//    auto input_type = to_data_type(tensor.type());
//    if (input_type != to_type)
//    {
//        add_convert(next_input, tensor, tf_id, to_type);
//    }
//    else
//    {
//        link_input_tensor(&next_input, tf_id);
//    }
//}