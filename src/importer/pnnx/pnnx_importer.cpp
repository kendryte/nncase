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
#include "pnnx_importer.h"
#include "ir.h"
#include <algorithm>
#include <cstdint>
#include <memory>
#include <nncase/importer/importer.h>
#include <nncase/importer/util.h>
#include <nncase/ir/graph.h>

using namespace std;
using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

pnnx_importer::pnnx_importer(std::span<const uint8_t> _paramfile, std::span<const uint8_t> _binfile, ir::graph &graph)
    : graph_(graph)
{
    paramfile = _paramfile;
    binfile = _binfile;
}

class FileWrapper
{
public:
    FileWrapper(const std::string &path)
    {
        fp = fopen(path.c_str(), "rb");
        if (!fp)
            throw std::runtime_error("Cannot open file: " + path);
    }

    ~FileWrapper()
    {
        fclose(fp);
    }

    FILE *fp;
};

void pnnx_importer::import(const struct import_options & /*options*/, std::string & /*real_inlayout*/, std::string & /*real_outlayout*/)
{
    // load param
    //     auto param_mem = (const unsigned char *)paramfile.data();
    //     auto bin_mem = (const unsigned char *)binfile.data();

    std::string parampath((const char *)paramfile.data());
    std::string binpath((const char *)binfile.data());

    pnnx::Graph pnnx_graph;
    pnnx_graph.load(parampath, binpath);

    for (const pnnx::Operator *op : pnnx_graph.ops)
    {
        convert_op(*op);
    }

    // connect tensors
    for (auto &&in : input_tensors_)
    {
        auto out_it = output_tensors_.find(in.second);
        if (out_it != output_tensors_.end())
        {
            in.first->connect(*out_it->second);
        }
        else
        {
            assert(!"Cannot find associated output node");
        }
    }

    // outputs
    for (auto &&out : output_tensors_)
    {
        if (out.second->connections().empty())
        {
            auto node = graph_.emplace<output_node>(out.second->type(), out.second->shape());
            node->name(out.first);
            out.second->connect(node->input());
        }
    }
}

void pnnx_importer::convert_op(const pnnx::Operator &op)
{
#define DEFINE_OPCODE(opcode, opcode2) \
    if (op.type == #opcode##sv)        \
        return convert_op_##opcode2(op);
#include "opcode.def"
#undef DEFINE_OPCODE

    throw std::runtime_error("Not supported pnnx opcode: " + op.type);
}
