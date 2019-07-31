#pragma once
#include <ir/quantizer.h>
#include <memory>
#include <scheduler/memory_allocator.h>
#include <transforms/transform.h>
#include <unordered_map>
#include <vector>

namespace nncase
{
class target
{
public:
    virtual void fill_allocators(std::unordered_map<memory_type_t, scheduler::memory_allocator *> &allocators, std::vector<std::unique_ptr<scheduler::memory_allocator>> &allocator_holders) = 0;
    virtual void registry_codegen_ops() = 0;
    virtual void registry_evaluator_ops() = 0;
    virtual void add_default_transforms(std::vector<std::unique_ptr<transforms::transform>> &transforms) = 0;
    virtual void add_optimize1_transforms(std::vector<std::unique_ptr<transforms::transform>> &transforms) = 0;
    virtual void add_optimize2_transforms(std::vector<std::unique_ptr<transforms::transform>> &transforms) = 0;
    virtual void add_quantization_checkpoint_transforms(std::vector<std::unique_ptr<transforms::transform>> &transforms) = 0;
    virtual void add_quantization_transforms(ir::quantizer& quantizer, std::vector<std::unique_ptr<transforms::transform>> &transforms) = 0;
};
}
