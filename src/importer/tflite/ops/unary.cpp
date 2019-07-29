#include "../tflite_importer.h"
#include <ir/ops/unary.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

DEFINE_TFLITE_LOWER(EXP)
{
    convert_unary(op, unary_exp);
}

DEFINE_TFLITE_LOWER(FLOOR)
{
    convert_unary(op, unary_floor);
}

DEFINE_TFLITE_LOWER(LOG)
{
    convert_unary(op, unary_log);
}

DEFINE_TFLITE_LOWER(NEG)
{
    convert_unary(op, unary_neg);
}

DEFINE_TFLITE_LOWER(RSQRT)
{
    convert_unary(op, unary_rsqrt);
}

DEFINE_TFLITE_LOWER(SIN)
{
    convert_unary(op, unary_sin);
}

void tflite_importer::convert_unary(const tflite::Operator &op, unary_op_t unary_op)
{
    auto &input = get_tensor(op.inputs(), 0);

    auto node = graph_.emplace<unary>(unary_op, get_shape(*input.shape()));

    input_tensors_.emplace(&node->input(), op.inputs()->Get(0));
    output_tensors_.emplace(op.outputs()->Get(0), &node->output());
}
