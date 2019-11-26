/* Copyright 2019 Canaan Inc.
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
#include "paddle_importer.h"
#include <fstream>
#include <importer/importer.h>
#include <ir/ops/constant.h>
#include <runtime/binary_reader.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;
using namespace nncase::runtime;
using namespace paddle::framework::proto;
using namespace std::string_view_literals;

paddle_importer::paddle_importer(xtl::span<const uint8_t> model, const boost::filesystem::path &params_dir, ir::graph &graph)
    : graph_(graph), subgraph_(nullptr), params_dir_(params_dir)
{
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    if (!model_.ParseFromArray(model.data(), (int)model.size()))
        throw std::runtime_error("Invalid PaddlePaddle model");
    subgraph_ = &model_.blocks()[0];
}

void paddle_importer::import()
{
    auto &operators = subgraph_->ops();
    for (auto &&op : operators)
        convert_op(op);
}

void paddle_importer::convert_op(const OpDesc &op)
{
    auto type = op.type();

#define DEFINE_OPCODE(opcode) \
    if (type == #opcode##sv)  \
        return convert_op_##opcode(op);
#include "opcode.def"
#undef DEFINE_OPCODE

    throw std::runtime_error("Not supported PaddlePaddle opcode: " + type);
}

const VarDesc &paddle_importer::find_var(std::string_view name) const
{
    auto &vars = subgraph_->vars();
    auto it = std::find_if(vars.begin(), vars.end(), [&](const VarDesc &var) { return var.name() == name; });
    if (it == vars.end())
        throw std::runtime_error("Variable named \"" + std::string(name) + "\" is not found");
    return *it;
}

const VarDesc &paddle_importer::find_var(const google::protobuf::RepeatedPtrField<OpDesc_Var> &container, std::string_view name) const
{
    auto &var_name = find_param(container, name).arguments()[0];
    return find_var(var_name);
}

void paddle_importer::load_tensor(std::string_view name, uint8_t *dest, uint8_t *end)
{
    auto filename = params_dir_ / std::string(name);
    std::ifstream infile(filename.string(), std::ios::binary | std::ios::in);
    if (!infile.good())
        throw std::runtime_error("Cannot open file: " + filename.string());

    binary_reader reader(infile);
    auto version = reader.read<uint32_t>();
    auto lod_level = reader.read<uint64_t>();
    for (uint64_t i = 0; i < lod_level; i++)
    {
        auto len = reader.read<uint64_t>();
        reader.skip(len);
    }

    version = reader.read<uint32_t>();
    if (version != 0)
        throw std::runtime_error("Unsupported tensor data version");
    auto desc_size = reader.read<uint32_t>();
    reader.skip(desc_size);

    auto avail = reader.avail();
    if (avail != (end - dest))
        throw std::invalid_argument("Unexpected tensor data size");
    reader.read_array(xtl::span<uint8_t> { dest, end });
}

const OpDesc_Var &paddle_importer::find_param(const google::protobuf::RepeatedPtrField<OpDesc_Var> &container, std::string_view name)
{
    auto it = std::find_if(container.begin(), container.end(), [&](const OpDesc_Var &var) { return var.parameter() == name; });
    if (it == container.end())
        throw std::runtime_error("Parameter named \"" + std::string(name) + "\" is not found");
    return *it;
}

const OpDesc_Attr &paddle_importer::find_attr(const google::protobuf::RepeatedPtrField<OpDesc_Attr> &container, std::string_view name)
{
    auto it = std::find_if(container.begin(), container.end(), [&](const OpDesc_Attr &var) { return var.name() == name; });
    if (it == container.end())
        throw std::runtime_error("Attribute named \"" + std::string(name) + "\" is not found");
    return *it;
}

shape_t paddle_importer::get_var_shape(const VarDesc &var)
{
    shape_t shape;
    auto &dims = var.type().lod_tensor().tensor().dims();
    for (auto d : dims)
    {
        if (d == -1)
            shape.push_back(1);
        else
            shape.push_back((size_t)d);
    }

    return shape;
}

graph nncase::importer::import_paddle(xtl::span<const uint8_t> model, const boost::filesystem::path &params_dir)
{
    graph graph;
    paddle_importer(model, params_dir, graph).import();
    return graph;
}
