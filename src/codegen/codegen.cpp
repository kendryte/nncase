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
#include <codegen/codegen.h>
#include <ir/op_utils.h>
#include <ir/ops/constant.h>
#include <runtime/model.h>
#include <string>
#include <unordered_map>
#include <unordered_set>

using namespace nncase;
using namespace nncase::codegen;
using namespace nncase::ir;
using namespace nncase::scheduler;
using namespace nncase::runtime;

namespace
{
std::unordered_map<node_opcode, emitter_t> g_emitters;
std::unordered_set<node_opcode> g_disabled_emitters;

std::unique_ptr<node_body> call_emitter(node &node, codegen_context &context)
{
    auto opcode = node.runtime_opcode();
    auto it = g_emitters.find(opcode);
    if (it == g_emitters.end())
    {
        if (g_disabled_emitters.find(opcode) == g_disabled_emitters.end())
            throw std::runtime_error(std::string("Emitter for ") + node_opcode_names(opcode).data() + " is not found");
    }
    else
    {
        return it->second(node, context);
    }

    return nullptr;
}
}

void nncase::codegen::register_emitter(node_opcode opcode, emitter_t emitter)
{
    g_emitters.emplace(opcode, std::move(emitter));
}

void nncase::codegen::disable_emitter(ir::node_opcode opcode)
{
    g_disabled_emitters.emplace(opcode);
}

codegen_context::codegen_context(std::ostream &output, const std::unordered_map<memory_type_t, memory_allocator *> &allocators, const std::unordered_map<ir::output_connector *, memory_allocation> &allocations)
    : writer_(output), allocators_(allocators), allocations_(allocations)
{
}

memory_range codegen_context::get_allocation(output_connector &conn) const
{
    auto &alloc = allocations_.at(&conn);
    return { alloc.type, conn.type(), (uint32_t)alloc.start, (uint32_t)alloc.size };
}

void nncase::codegen::gencode(codegen_context &context, xtl::span<ir::node *> compute_sequence)
{
    std::vector<ir::node *> runtime_nodes;
    std::vector<memory_range> inputs;
    std::vector<runtime_shape_t> input_shapes;
    std::vector<memory_range> outputs;
    std::vector<ir::node *> constants;

    for (auto &&node : compute_sequence)
    {
        if (g_disabled_emitters.find(node->runtime_opcode()) == g_disabled_emitters.end())
            runtime_nodes.emplace_back(node);

        switch (node->runtime_opcode())
        {
        case op_input_node:
            inputs.emplace_back(context.get_allocation(node->output_at(0)));
            input_shapes.emplace_back(ir::to(node->output_at(0).shape()));
            break;
        case op_output_node:
            outputs.emplace_back(context.get_allocation(*node->input_at(0).connection()));
            break;
        case op_constant:
            constants.emplace_back(node);
            break;
        }
    }

    auto &writer = context.writer();
    // model header
    model_header model_header;
    model_header.identifier = MODEL_IDENTIFIER;
    model_header.version = MODEL_VERSION;
    model_header.flags = 0;
    model_header.target = MODEL_TARGET_K210;
    model_header.constants = context.constant_usage();
    model_header.main_mem = context.memory_usage();
    model_header.nodes = runtime_nodes.size();
    model_header.inputs = inputs.size();
    model_header.outputs = outputs.size();

    writer.write(model_header);

    // inputs
    writer.write_array<memory_range>(inputs);
    // input shapes
    writer.write_array<runtime_shape_t>(input_shapes);
    // outputs
    writer.write_array<memory_range>(outputs);

    // constants
    auto const_mem = std::make_unique<uint8_t[]>(context.constant_usage());
    for (auto &node : constants)
    {
        auto &con = static_cast<constant &>(*node);
        auto alloc = context.get_allocation(con.output());
        auto start = const_mem.get() + alloc.start;
        std::copy(con.data().begin(), con.data().end(), start);
    }

    writer.write_array(xtl::span<const uint8_t> { const_mem.get(), context.constant_usage() });

    // Keep node headers
    std::vector<node_header> node_headers;
    auto node_headers_pos = writer.position();
    std::streamoff node_header_bytes = sizeof(node_header) * runtime_nodes.size();

    writer.position(node_headers_pos + node_header_bytes);

    // write body
    for (auto &&node : runtime_nodes)
    {
        auto body = call_emitter(*node, context);
        if (body)
        {
            auto body_start = writer.position();
            body->serialize(writer);
            writer.align_position(8);
            auto body_size = writer.position() - body_start;
            node_headers.emplace_back(node_header { body->opcode(), (uint32_t)body_size });
        }
    }

    // Write node headers
    auto end_pos = writer.position();
    writer.position(node_headers_pos);
    writer.write_array<node_header>(node_headers);
    writer.position(end_pos);

    std::cout << "Main memory usage: " << context.memory_usage() << " B" << std::endl;
}
