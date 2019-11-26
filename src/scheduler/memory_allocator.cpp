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
#include <ir/op_utils.h>
#include <ortools/sat/cp_model.h>
#include <ortools/sat/cp_model_solver.h>
#include <scheduler/freelist.h>
#include <scheduler/memory_allocator.h>
#include <stdexcept>

using namespace nncase;
using namespace nncase::scheduler;
using namespace operations_research;
using namespace operations_research::sat;

namespace
{
constexpr size_t align(size_t size, size_t alignment = 8)
{
    size_t remainder = size % alignment;
    if (remainder != 0)
        return size - remainder + alignment;
    return size;
}

struct memory_node_var
{
    IntervalVar x;
    IntervalVar y;
};

memory_node_var add_memory_node_var(int32_t index, CpModelBuilder &model, size_t alignment, const memory_node &node, NoOverlap2DConstraint &no_overlap)
{
    Domain memory_loc(0, std::numeric_limits<uint32_t>::max());

    auto prefix = "node_" + std::to_string(index) + "_";
    // X: duration (constant)
    auto x = model.NewIntervalVar(model.NewConstant(node.birth()), model.NewConstant(node.age()), model.NewConstant(node.birth() + node.age()));
    // Y: location (variable)
    auto y = model.NewIntervalVar(model.NewIntVar(memory_loc).WithName(prefix + "y_start"), model.NewConstant(node.size() / alignment), model.NewIntVar(memory_loc).WithName(prefix + "y_end"));
    no_overlap.AddRectangle(x, y);
    return { x, y };
}
}

memory_node::memory_node(memory_allocator &allocator, size_t birth, size_t valid_size, size_t size)
    : allocator_(allocator), birth_(birth), valid_size_(valid_size), size_(size), use_count_(0), age_(0)
{
}

void memory_node::add_ref()
{
    ++use_count_;
}

void memory_node::release()
{
    if (--use_count_ == 0)
        allocator_.free(*this);

    if (use_count_ < 0)
        throw std::runtime_error("Memory node has been freed");
}

void memory_node::grow_age()
{
    age_++;
}

memory_allocator::memory_allocator(size_t alignment, std::optional<size_t> fixed_size)
    : alignment_(alignment), age_(0), fixed_size_(fixed_size)
{
}

memory_node &memory_allocator::allocate(size_t size)
{
    auto aligned_size = align(size, alignment_);
    //auto free_node = freelist_.allocate(aligned_size);
    auto &node = nodes_.emplace_back(*this, age_, size, aligned_size);
    node.add_ref();
    return node;
}

void memory_allocator::free(memory_node &node)
{
    //freelist_.free(free_memory_node { node.start(), node.size() });
}

void memory_allocator::grow_age()
{
    age_++;
    for (auto &n : nodes_)
    {
        if (n.used())
            n.grow_age();
    }
}

void memory_allocator::finish(uint32_t max_solve_secs)
{
    bool solved = false;
    size_t max_usage = 0;

    if (max_solve_secs != 0)
    {
        CpModelBuilder builder;
        auto no_overlap = builder.AddNoOverlap2D();

        std::vector<memory_node_var> node_vars;
        node_vars.reserve(nodes_.size());
        int32_t index = 0;
        for (auto &n : nodes_)
            node_vars.emplace_back(add_memory_node_var(index++, builder, alignment_, n, no_overlap));

        std::vector<IntVar> all_end_vars;
        all_end_vars.reserve(nodes_.size());
        for (auto &v : node_vars)
            all_end_vars.emplace_back(v.y.EndVar());

        auto cost = LinearExpr::Sum(all_end_vars);
        builder.Minimize(cost);

        Model model;
        // Tell the solver to enumerate all solutions.
        SatParameters parameters;
        parameters.set_find_multiple_cores(true);
        parameters.set_max_time_in_seconds(max_solve_secs);
        model.Add(NewSatParameters(parameters));
        auto r = SolveCpModel(builder.Build(), &model);
        auto status = r.status();
        if (status != CpSolverStatus::FEASIBLE
            && status != CpSolverStatus::OPTIMAL)
        {
            std::cout << "  Allocator cannot solve a optimal layout in " << parameters.max_time_in_seconds()
                      << "secs, use the first fit method instead." << std::endl;
        }
        else
        {
#if 0
            index = 0;
            for (auto &v : node_vars)
            {
                LOG(INFO) << "node_" << index++ << "_y_start = " << SolutionIntegerValue(r, v.y.StartVar());
                LOG(INFO) << "node_" << index << "_y_size = " << SolutionIntegerValue(r, v.y.SizeVar());
                LOG(INFO) << "node_" << index << "_y_end = " << SolutionIntegerValue(r, v.y.EndVar());
            }
#endif
            index = 0;
            for (auto &n : nodes_)
            {
                auto &v = node_vars[index++];
                n.start(SolutionIntegerValue(r, v.y.StartVar()) * alignment_);
                max_usage = std::max(max_usage, (size_t)SolutionIntegerValue(r, v.y.EndVar()) * alignment_);
            }

            solved = true;
        }
    }

    if (!solved)
    {
        freelist fl(std::nullopt);
        size_t age = 0;
        size_t allocated_nodes = 0;
        while (allocated_nodes < nodes_.size())
        {
            for (auto &n : nodes_)
            {
                if (age == n.birth())
                {
                    auto alloc_node = fl.allocate(n.size());
                    n.start(alloc_node.start);
                    allocated_nodes++;
                }

                if (age == n.birth() + n.age())
                {
                    fl.free({ n.start(), n.size() });
                }
            }

            age++;
        }

        max_usage = fl.max_usage();
    }

    if (fixed_size_ && max_usage > *fixed_size_)
        throw std::runtime_error("KPU allocator cannot allocate more memory.");
    max_usage_ = max_usage;
}

size_t memory_allocator::get_bytes(datatype_t type, const ir::shape_t &shape) const
{
    return runtime::get_bytes(type) * xt::compute_size(shape);
}
