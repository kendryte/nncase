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

module_t tflite_importer::import(const import_options &options) {
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

    // 3. Create function
    function main_func("main", created_inputs, expr(nullptr));
    module_->functions().emplace_back(main_func);
    module_->entry(main_func);

    // std::unordered_map<int32_t, input_connector *> created_outputs;

    // create inputs
    // for (auto &&in : *subgraph_->inputs())
    //{
    //    auto &tensor = *subgraph_->tensors()->Get(in);
    //    auto type = get_ir_type(tensor.shape(), tensor.type());
    //    auto shape = get_shape(tensor.shape());
    //    auto type = to_data_type(tensor.type());
    //    // image
    //    if (options.input_layout == "NCHW" && shape.size() == 4)
    //    {
    //        auto trans = nhwc_to_nchw(shape);
    //        auto node = graph_.emplace<input_node>(type, trans);
    //        node->name(tensor.name()->string_view());
    //        auto sur_trans = nchw_to_nhwc(node->output().type(),
    //        node->output().shape());
    //        sur_trans->name(tensor.name()->string_view());
    //        sur_trans->input().connect(node->output());
    //        created_inputs.emplace(in, &sur_trans->output());
    //    }
    //    else
    //    {
    //        auto node = graph_.emplace<input_node>(type, shape);
    //        node->name(tensor.name()->string_view());
    //        created_inputs.emplace(in, &node->output());
    //    }
    //}

    // auto &operators = *subgraph_->operators();
    // for (auto &&op : operators)
    //    convert_op(*op);

    // std::vector<int32_t> outputs;
    // if (options.output_arrays.empty())
    //{
    //    for (auto &&out : *subgraph_->outputs())
    //    {
    //        outputs.emplace_back(out);
    //    }
    //}
    // else
    //{
    //    for (auto &&name : options.output_arrays)
    //    {
    //        bool found = false;
    //        size_t i = 0;
    //        for (auto &&t : *subgraph_->tensors())
    //        {
    //            auto t_name = t->name();
    //            if (t_name && t_name->string_view() == name)
    //            {
    //                outputs.emplace_back(i);
    //                found = true;
    //                break;
    //            }

    //            i++;
    //        }

    //        if (!found)
    //        {
    //            throw std::runtime_error("Cannot find output tensor: " +
    //            name);
    //        }
    //    }
    //}

    // create outputs
    // for (auto &&out : outputs)
    //{
    //    auto &tensor = *subgraph_->tensors()->Get(out);
    //    auto shape = get_shape(tensor.shape());
    //    auto type = to_data_type(tensor.type());
    //    // image
    //    if (options.output_layout == "NCHW" && shape.size() == 4)
    //    {
    //        auto pre_trans = nhwc_to_nchw(type, shape);
    //        pre_trans->name(tensor.name()->string_view());
    //        auto node =
    //        graph_.emplace<output_node>(pre_trans->output().type(),
    //        pre_trans->output().shape());
    //        node->name(tensor.name()->string_view());
    //        node->input().connect(pre_trans->output());
    //        created_outputs.emplace(out, &pre_trans->input());
    //    }
    //    else
    //    {
    //        auto node = graph_.emplace<output_node>(type, shape);
    //        node->name(tensor.name()->string_view());
    //        created_outputs.emplace(out, &node->input());
    //    }
    //}

    // connect tensors
    // for (auto &&in : input_tensors_)
    //{
    //    auto out_it = output_tensors_.find(in.second);
    //    if (out_it != output_tensors_.end())
    //    {
    //        in.first->connect(*out_it->second);
    //    }
    //    else
    //    {
    //        auto &tensor = *subgraph_->tensors()->Get(in.second);
    //        auto &buffer = *model_->buffers()->Get(tensor.buffer());
    //        auto data = buffer.data();

    //        if (data)
    //        {
    //            auto type = to_data_type(tensor.type());
    //            auto shape = get_shape(tensor.shape());
    //            auto con = graph_.emplace<constant>(type, shape,
    //            std::as_bytes(std::span(data->data(), data->data() +
    //            data->size()))); con->name(tensor.name()->string_view());
    //            link_output_tensor(in.second, &con->output());
    //            in.first->connect(con->output());
    //        }
    //    }
    //}

    // inputs
    // for (auto &&in : input_tensors_)
    //{
    //    if (!in.first->connection())
    //    {
    //        auto out = created_inputs.at(in.second);
    //        in.first->connect(*out);
    //    }
    //}

    // outputs
    // for (auto &&out : output_tensors_)
    //{
    //    auto in = created_outputs.find(out.first);
    //    if (in != created_outputs.end())
    //    {
    //        in->second->connect(*out.second);
    //    }
    //}

    // outputs that connect to inputs or constants
    // for (auto &&out : created_outputs)
    //{
    //    if (!out.second->connection())
    //    {
    //        auto &tensor = *subgraph_->tensors()->Get(out.first);
    //        auto &buffer = *model_->buffers()->Get(tensor.buffer());
    //        auto data = buffer.data();

    //        if (data)
    //        {
    //            auto type = to_data_type(tensor.type());
    //            auto shape = get_shape(tensor.shape());
    //            auto con = graph_.emplace<constant>(type, shape,
    //            std::as_bytes(std::span(data->data(), data->data() +
    //            data->size()))); con->name(tensor.name()->str() + "/const");
    //            out.second->connect(con->output());
    //        }
    //        else
    //        {
    //            auto in = created_inputs.find(out.first);
    //            if (in != created_inputs.end())
    //                out.second->connect(*in->second);
    //        }
    //    }
    //}

    return module_;
}

ir::type tflite_importer::get_ir_type(const flatbuffers::Vector<int32_t> *shape,
                                      tflite::TensorType tflite_type) {
    shape_t ir_shape;
    if (shape) {
        std::ranges::transform(*shape, std::back_inserter(ir_shape),
                               [](int32_t dim) -> dim_t { return dim; });
    }
    return tensor_type(to_datatype(tflite_type), ir_shape);
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
    return output_tensors_.at(id);
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