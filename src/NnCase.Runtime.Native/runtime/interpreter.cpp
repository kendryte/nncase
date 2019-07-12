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
    input_shapes_ = { reinterpret_cast<const runtime_shape_t *>(offset), inputs_size() };

    return true;
}

void interpreter::run(run_callback_t callback)
{
}
