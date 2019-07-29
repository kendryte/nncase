#include "../tflite_importer.h"
#include <ir/ops/binary.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

DEFINE_TFLITE_LOWER(ADD)
{
    convert_binary(op, binary_add, op.builtin_options_as_AddOptions()->fused_activation_function());
}

DEFINE_TFLITE_LOWER(DIV)
{
    convert_binary(op, binary_div, op.builtin_options_as_DivOptions()->fused_activation_function());
}

DEFINE_TFLITE_LOWER(MAXIMUM)
{
    convert_binary(op, binary_max, tflite::ActivationFunctionType_NONE);
}

DEFINE_TFLITE_LOWER(MINIMUM)
{
    convert_binary(op, binary_min, tflite::ActivationFunctionType_NONE);
}

DEFINE_TFLITE_LOWER(MUL)
{
    convert_binary(op, binary_mul, op.builtin_options_as_MulOptions()->fused_activation_function());
}

DEFINE_TFLITE_LOWER(SUB)
{
    convert_binary(op, binary_sub, op.builtin_options_as_SubOptions()->fused_activation_function());
}

void tflite_importer::convert_binary(const tflite::Operator &op, binary_op_t binary_op, tflite::ActivationFunctionType activation)
{
    auto &input_a = get_tensor(op.inputs(), 0);
    auto &input_b = get_tensor(op.inputs(), 1);

    auto add = graph_.emplace<binary>(binary_add, get_shape(*input_a.shape()), get_shape(*input_b.shape()), to_float_clamp_range(activation));

    input_tensors_.emplace(&add->input_a(), op.inputs()->Get(0));
    input_tensors_.emplace(&add->input_b(), op.inputs()->Get(1));
    output_tensors_.emplace(op.outputs()->Get(0), &add->output());
}
