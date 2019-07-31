#pragma once
#include <targets/target.h>
#include "..//cpu/target.h"

namespace nncase
{
class k210_target : public cpu_target
{
public:
    void fill_allocators(std::unordered_map<memory_type_t, scheduler::memory_allocator *> &allocators, std::vector<std::unique_ptr<scheduler::memory_allocator>> &allocator_holders) override;
    void registry_codegen_ops() override;
    void registry_evaluator_ops() override;
    void add_default_transforms(std::vector<std::unique_ptr<transforms::transform>> &transforms) override;
    void add_optimize1_transforms(std::vector<std::unique_ptr<transforms::transform>> &transforms) override;
    void add_optimize2_transforms(std::vector<std::unique_ptr<transforms::transform>> &transforms) override;
    void add_quantization_checkpoint_transforms(std::vector<std::unique_ptr<transforms::transform>> &transforms) override;
    void add_quantization_transforms(ir::quantizer &quantizer, std::vector<std::unique_ptr<transforms::transform>> &transforms) override;
};
}
