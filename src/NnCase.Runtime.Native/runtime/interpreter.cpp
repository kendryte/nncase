#include <runtime/interpreter.h>

using namespace nncase;
using namespace nncase::runtime;

bool interpreter::try_load_model(const uint8_t *buffer)
{
    auto offset = buffer;
    model_header_ = reinterpret_cast<const model_header *>(buffer);

    // Validate model
    if (model_header_->identifier != MODEL_IDENTIFIER || model_header_->version != MODEL_VERSION || (model_header_->target != MODEL_TARGET_CPU && model_header_->target != MODEL_TARGET_K210))
        return false;

    // Allocate buffers
    main_mem_.reset(new (std::nothrow) uint8_t[model_header_->main_mem]);
    if (!main_mem_)
        return false;

    offset += sizeof(model_header);
    inputs_ = { reinterpret_cast<const memory_range *>(offset), inputs_size() };
    offset += sizeof(memory_range) * inputs_size();
    input_shapes_ = { reinterpret_cast<const runtime_shape_t *>(offset), inputs_size() };
    offset += sizeof(runtime_shape_t) * inputs_size();
    outputs_ = { reinterpret_cast<const memory_range *>(offset), outputs_size() };
    offset += sizeof(memory_range) * outputs_size();
    constants_ = { offset, model_header_->constants };
    offset += constants_.size();
    node_headers_ = { reinterpret_cast<const node_header *>(offset), nodes_size() };
    offset += sizeof(node_header) * nodes_size();
    node_body_start_ = offset;

    return true;
}

void interpreter::run(run_callback_t callback)
{
    run_callback_ = callback;

}

xtl::span<uint8_t> interpreter::memory_at(const memory_range &range) const noexcept
{
    uintptr_t base;

    switch (range.memory_type)
    {
    case mem_const:
        base = (uintptr_t)constants_.data();
        break;
    case mem_main:
        base = (uintptr_t)main_mem_.get();
        break;
    default:
        base = 0;
        break;
    }

    return { reinterpret_cast<uint8_t *>(base + range.start), range.size };
}
