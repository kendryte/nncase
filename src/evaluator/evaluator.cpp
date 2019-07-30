#include <chrono>
#include <ir/evaluator.h>
#include <ir/ops/constant.h>
#include <ir/quantizer.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::scheduler;
namespace chrono = std::chrono;

namespace
{
std::unordered_map<node_opcode, std::function<void(ir::node &, evaluate_context &)>> g_evaluators;

auto &get_evaluator(node_opcode opcode)
{
    auto it = g_evaluators.find(opcode);
    if (it == std::end(g_evaluators))
        throw std::runtime_error("Evaluator for " + std::string(node_opcode_names(opcode)) + " is not found");
    return it->second;
}
}

void nncase::ir::register_evaluator(ir::node_opcode opcode, std::function<void(ir::node &, evaluate_context &)> evaluator)
{
    g_evaluators.emplace(opcode, std::move(evaluator));
}

evaluate_context::evaluate_context(const std::unordered_map<memory_type_t, memory_allocator *> &allocators, const std::unordered_map<ir::output_connector *, memory_allocation> &allocations)
    : allocators_(allocators), allocations_(allocations)
{
    for (auto &&allocator : allocators_)
    {
        memory_pools_.emplace(allocator.first, std::make_unique<uint8_t[]>(allocator.second->max_usage()));
    }
}

xtl::span<uint8_t> evaluate_context::memory_at(const scheduler::memory_allocation &allocation)
{
    auto& memory_pool = memory_pools_.at(allocation.type);

    return { memory_pool.get() + allocation.start, allocation.size };
}

evaluator::evaluator(evaluate_context &context, xtl::span<ir::node *> compute_sequence)
    : context_(context), compute_sequence_(compute_sequence)
{
    for (auto &&node : compute_sequence_)
    {
        switch (node->runtime_opcode())
        {
        case op_input_node:
            inputs_.emplace_back(&node->output_at(0));
            break;
        case op_output_node:
            outputs_.emplace_back(&node->input_at(0));
            break;
        case op_constant:
        {
            auto &rnode = static_cast<constant &>(*node);
            auto src = rnode.data();
            std::copy(std::begin(src), std::end(src), context.memory_at<uint8_t>(rnode.output()).begin());
            break;
        }
        }
    }
}

void evaluator::evaluate(quantizer *quantizer)
{
    using clock = chrono::high_resolution_clock;
    chrono::nanoseconds total_duration = {};

    for (auto &&node : compute_sequence_)
    {
        auto &evaluator = get_evaluator(node->runtime_opcode());

        auto start = clock::now();
        evaluator(*node, context_);
        auto duration = clock::now() - start;
        total_duration += duration;
        std::cout << node_opcode_names(node->runtime_opcode()) << ": " << duration.count() / 1e6 << "ms" << std::endl;

        if (quantizer && node->runtime_opcode())
        {
            auto opcode = node->runtime_opcode();
            if (opcode == op_fake_dequantize || opcode == op_fake_quantize)
            {
                auto &output = node->output_at(0);
                quantizer->record(output, context_.memory_at<float>(output));
            }
        }
    }

    std::cout << "Total: " << total_duration.count() / 1e6 << "ms" << std::endl;
}
