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
#include <fstream>
#include <nncase/ir/debug.h>
#include <nncase/ir/visitor.h>
#include <nncase/version.h>
#include <onnx.pb.h>
#include <ostream>
#include <queue>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/transform.hpp>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace nncase;
using namespace nncase::ir;
using namespace google::protobuf;

namespace {
std::unordered_map<datatype_t, int32> onnx_types_map{
    {dt_int8, onnx::TensorProto_DataType_INT8},
    {dt_int16, onnx::TensorProto_DataType_INT16},
    {dt_int32, onnx::TensorProto_DataType_INT32},
    {dt_int64, onnx::TensorProto_DataType_INT64},
    {dt_uint8, onnx::TensorProto_DataType_UINT8},
    {dt_uint16, onnx::TensorProto_DataType_UINT16},
    {dt_uint32, onnx::TensorProto_DataType_UINT32},
    {dt_uint64, onnx::TensorProto_DataType_UINT64},
    {dt_float16, onnx::TensorProto_DataType_FLOAT16},
    {dt_float32, onnx::TensorProto_DataType_FLOAT},
    {dt_float64, onnx::TensorProto_DataType_DOUBLE},
    {dt_bfloat16, onnx::TensorProto_DataType_BFLOAT16},
    {dt_bfloat16, onnx::TensorProto_DataType_BOOL},
    {dt_string, onnx::TensorProto_DataType_STRING}};

int32 to_pb(datatype_t dt) {
    assert(onnx_types_map.contains(dt));
    return onnx_types_map.at(dt);
}

void to_pb([[maybe_unused]] onnx::TypeProto_Tensor *dst,
           [[maybe_unused]] const type &src) {
    // dst->set_elem_type(to_pb(src.elem_type()));
    // if (src.shape().is_fixed() || src.shape().has_unknown_dim()) {
    //    auto shape_proto = dst->mutable_shape();
    //    for (auto dim : src.shape()) {
    //        shape_proto->add_dim()->set_dim_value(dim.is_fixed() ? dim.value
    //                                                             : -1);
    //    }
    //}
}

void dump_function_pb(const ir::function &func,
                      const std::filesystem::path &dst_path) {

    GOOGLE_PROTOBUF_VERIFY_VERSION;

    onnx::ModelProto model;
    model.set_producer_name("nncase ir");
    model.set_producer_version(NNCASE_VERSION);

    auto gp = model.mutable_graph();
    gp->set_name(func->name());

    // 1. inputs
    for (auto in : func->parameters()) {
        auto inp = gp->add_input();
        inp->set_name(in->name());
        auto type = inp->mutable_type();
        to_pb(type->mutable_tensor_type(), in->type_annotation());
    }

    // 2. outputs
    // for (auto out : src_graph.outputs()) {
    //    auto outp = gp->add_output();
    //    outp->set_name(out->name());
    //    auto type = outp->mutable_type();
    //    auto ttype = type->mutable_tensor_type();
    //    ttype->set_elem_type(to_pb(out->input().type()));
    //    to_pb(ttype->mutable_shape(), out->input().shape());
    //}

    // 3. nodes
    /*for (auto &n : src_graph.nodes()) {
        if (n->runtime_opcode() != op_input_node &&
            n->runtime_opcode() != op_output_node) {
            auto np = gp->add_node();
            np->set_name(n->name());
            np->set_op_type(std::string(n->runtime_opcode().name));

            if (auto c = node_cast<constant>(*n)) {
                auto valuep = np->add_attribute();
                valuep->set_name("value");
                valuep->set_type(onnx::AttributeProto_AttributeType_TENSOR);
                auto tp = valuep->mutable_t();
                tp->set_name(c->name());
                tp->set_data_type(to_pb(c->output().type()));
                for (auto dim : c->output().shape())
                    tp->add_dims((int64)dim);
                tp->set_raw_data(c->data().data(), c->data().size());
            } else {
                auto att_md = np->add_attribute();
                att_md->set_name("module_type");
                att_md->add_strings(n->module_type().data());

                auto att_act = np->add_attribute();
                att_act->set_name("action");
                att_act->add_strings(
                    (n->attributes() & node_attr_action) ? "true" : "false");
            }

            for (auto in : n->inputs()) {
                auto out = in->connection();
                if (out->owner().outputs().size() == 1)
                    np->add_input(out->owner().name());
                else
                    np->add_input(out->owner().name() + ":" + out->name());
            }

            for (auto out : n->outputs()) {
                std::string name;
                for (auto in : out->connections()) {
                    if (in->owner().runtime_opcode() == op_output_node)
                        name = in->owner().name();
                }

                if (name.empty()) {
                    if (n->outputs().size() == 1)
                        name = n->name();
                    else
                        name = n->name() + ":" + out->name();
                }

                np->add_output(name);
                auto vi = gp->add_value_info();
                vi->set_name(name);
                auto type = vi->mutable_type();
                auto ttype = type->mutable_tensor_type();
                ttype->set_elem_type(to_pb(out->type()));
                to_pb(ttype->mutable_shape(), out->shape());
            }
        }
    }*/

    auto filename = dst_path / (func->name() + ".nnir.pb");
    auto dirname = filename.parent_path();
    if (!std::filesystem::exists(dirname))
        std::filesystem::create_directories(dirname);
    std::ofstream ofile(filename, std::ios::out | std::ios::binary);
    model.SerializeToOstream(&ofile);
}

class il_dump_visitor : public expr_functor<std::string> {
  public:
    using expr_functor::visit;

    std::string visit(const function &ex) override {
        std::stringstream ss;
        // 1. Function signature
        {
            in_function_body_ = false;
            auto name = "%" + ex->name();
            names_.emplace(ex.get(), name);
            ident(ss) << name << " = fn (";
            size_t i = 0;
            for (auto &par : ex->parameters()) {
                ss << visit(par);
                if (++i != ex->parameters().size())
                    ss << ", ";
            }
            ss << ") {" << std::endl;
        }

        // 2. Function body
        {
            in_function_body_ = true;
            ident_level_++;
            ss << visit(ex->body());
            ident_level_--;
        }

        // 3. Function closing
        ident(ss) << "}" << std::endl;
        return ss.str();
    }

    std::string visit(const var &ex) override {
        std::stringstream ss;
        auto name = "%" + ex->name();
        names_.emplace(ex.get(), name);
        if (in_function_body_) {
            ident(ss) << name << std::endl;
        } else {
            ss << name;
            // if (!ex->type_annotation().empty())
            //    visit_type(ex->type_annotation());
        }
        return ss.str();
    }

    std::string visit(const call &ex) override {
        auto it = names_.find(ex.get());
        if (it != names_.end()) {
            return it->second;
        } else {
            std::stringstream ss;
            auto name = "%" + std::to_string(local_id_++);
            names_.emplace(ex.get(), name);
            ident(ss) << " = " << expr_functor::visit(ex->target());
            temp_vars_.emplace(ss.str());
            return name;
        }
    }

    // std::string visit_type(const type &t) override {
    //    os_ << ": "
    //        << "any" /*to_string(t)*/;
    //}

  private:
    std::ostream &ident(std::ostream &os) {
        for (size_t i = 0; i < ident_level_; i++)
            os << "  ";
        return os;
    }

  private:
    size_t ident_level_ = 0;
    bool in_function_body_;
    std::unordered_map<expr_node *, std::string> names_;
    std::stack<std::string> temp_vars_;
    size_t local_id_ = 0;
};
void dump_function_il(const ir::function &func,
                      const std::filesystem::path &dst_path) {
    auto filename = dst_path / (func->name() + ".nnir.il");
    auto dirname = filename.parent_path();
    if (!std::filesystem::exists(dirname))
        std::filesystem::create_directories(dirname);

    std::ofstream ofile(filename, std::ios::out);
    il_dump_visitor dumper;
    ofile << dumper(func);
}
} // namespace

void ir::dump_function(const ir::function &func,
                       const std::filesystem::path &dst_path) {
    dump_function_pb(func, dst_path);
    // dump_function_il(func, dst_path);
}
