#include "RuntimeTensor.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <nncase/runtime/interpreter.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/version.h>
#include <type_traits>
#include <vector>

using namespace nncase;
using namespace nncase::runtime;

static std::unique_ptr<interpreter> _interp;

/**
 * @brief init the interpreter
 *
 */
EXPORT_API(bool)
interpreter_init() {
    if (!_interp) {
        _interp = std::make_unique<nncase::runtime::interpreter>();
        return true;
    }
    return false;
}

/**
 * @brief load model
 *
 * @param buffer_ptr the buffer array
 * @param size buffer length
 */
EXPORT_API(void)
interpreter_load_model(uint8_t *buffer_ptr, int size) {
    auto buffer =
        std::span<const std::byte>((const std::byte *)(buffer_ptr), size);
    _interp->load_model(buffer).unwrap_or_throw();
}

/**
 * @brief get the model inputs size
 *
 * @return size_t
 */
EXPORT_API(size_t)
interpreter_inputs_size() { return _interp->inputs_size(); }

/**
 * @brief get the model outputs size
 *
 * @return size_t
 */
EXPORT_API(size_t)
interpreter_outputs_size() { return _interp->outputs_size(); }
/**
 * @brief get the input memory range desc
 *
 * @param index input number
 * @return memory_range
 */
EXPORT_API(memory_range)
interpreter_get_input_desc(size_t index) { return _interp->input_desc(index); }
/**
 * @brief get the output memory range desc
 *
 * @param index output number
 * @return memory_range
 */
EXPORT_API(memory_range)
interpreter_get_output_desc(size_t index) {
    return _interp->output_desc(index);
}

/**
 * @brief get the input tensor impl pointer
 *
 * @param index input number
 * @return void* the runtime_tensor impl pointer
 */
EXPORT_API(runtime_tensor *)
interpreter_get_input_tensor(size_t index) {
    return new runtime_tensor(
        std::move(_interp->input_tensor(index).unwrap_or_throw()));
}

/**
 * @brief set the input tensor
 *
 * @param index
 * @param rt
 */
EXPORT_API(void)
interpreter_set_input_tensor(size_t index, runtime_tensor *rt) {
    _interp->input_tensor(index, *rt).unwrap_or_throw();
}

EXPORT_API(runtime_tensor *)
interpreter_get_output_tensor(size_t index) {
    return new runtime_tensor(
        std::move(_interp->output_tensor(index).unwrap_or_throw()));
}

/**
 * @brief set the output tensor
 *
 * @param index
 * @param rt
 */
EXPORT_API(void)
interpreter_set_output_tensor(size_t index, runtime_tensor *rt) {
    _interp->input_tensor(index, *rt).unwrap_or_throw();
}

/**
 * @brief call the interpreter
 *
 */
EXPORT_API(void)
interpreter_run() { _interp->run().unwrap_or_throw(); }
