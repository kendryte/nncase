#include "target.h"
#include <scheduler/main_memory_allocator.h>
#include <transforms/neutral/fold_pad.h>
#include <transforms/neutral/fold_quantize.h>
#include <transforms/neutral/fold_transpose.h>
#include <transforms/neutral/fold_reshape.h>
#include <transforms/neutral/fuse_pad.h>
#include <transforms/neutral/transpose_motion.h>

using namespace nncase;
using namespace nncase::scheduler;
using namespace nncase::transforms;

namespace nncase
{
namespace codegen
{
    void register_netural_emitters();
}
}

namespace nncase
{
namespace ir
{
    void register_neutral_evaluators();
}
}

void nncase::cpu_target::fill_allocators(std::unordered_map<memory_type_t, scheduler::memory_allocator *> &allocators, std::vector<std::unique_ptr<memory_allocator>> &allocator_holders)
{
    allocators.emplace(mem_const, allocator_holders.emplace_back(std::make_unique<main_memory_allocator>()).get());
    allocators.emplace(mem_main, allocator_holders.emplace_back(std::make_unique<main_memory_allocator>()).get());
}

void nncase::cpu_target::registry_codegen_ops()
{
    using namespace nncase::codegen;

    register_netural_emitters();
}

void nncase::cpu_target::registry_evaluator_ops()
{
    using namespace nncase::ir;

    register_neutral_evaluators();
}

void nncase::cpu_target::add_default_transforms(std::vector<std::unique_ptr<transform>> &transforms)
{
    transforms.emplace_back(new fold_nop_pad_transform());
    transforms.emplace_back(new fold_nop_reshape_transform());
    transforms.emplace_back(new fold_nop_transpose_transform());
    transforms.emplace_back(new fold_pad_strided_slice_transform());
    transforms.emplace_back(new fold_quantize_transform());
    transforms.emplace_back(new fold_reshape_transform());
    transforms.emplace_back(new fold_transpose_transform());
    transforms.emplace_back(new fuse_pad_conv2d_transform());
    transforms.emplace_back(new strided_slice_to_pad_transform());
    transforms.emplace_back(new transpose_binary_motion_transform());
    transforms.emplace_back(new transpose_constant_binary_motion_transform());
    transforms.emplace_back(new transpose_concat_motion_transform());
    transforms.emplace_back(new transpose_pad_motion_transform());
    transforms.emplace_back(new transpose_reduce_motion_transform());
    transforms.emplace_back(new transpose_to_reshape_transform());
}

void nncase::cpu_target::add_optimize1_transforms(std::vector<std::unique_ptr<transform>> &transforms)
{
}

void nncase::cpu_target::add_optimize2_transforms(std::vector<std::unique_ptr<transform>> &transforms)
{
}

void nncase::cpu_target::add_quantization_checkpoint_transforms(std::vector<std::unique_ptr<transform>> &transforms)
{
}

void nncase::cpu_target::add_quantization_transforms(ir::quantizer &quantizer, std::vector<std::unique_ptr<transform>> &transforms)
{
}
