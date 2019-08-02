#include "../tflite_importer.h"
#include <ir/ops/binary.h>
#include <ir/ops/constant.h>
#include <ir/ops/reduce.h>
#include <ir/ops/unary.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

DEFINE_TFLITE_LOWER(LEAKY_RELU)
{
    auto &input = get_tensor(op.inputs(), 0);
    auto &options = *op.builtin_options_as_LeakyReluOptions();
    auto in_shape = get_shape(*input.shape());

    auto alpha = graph_.emplace<constant>(options.alpha());
    auto mul = graph_.emplace<binary>(binary_mul, in_shape, alpha->output().shape(), value_range<float>::full());
    auto max = graph_.emplace<binary>(binary_max, in_shape, mul->output().shape(), value_range<float>::full());

    mul->input_b().connect(alpha->output());
    max->input_b().connect(mul->output());

    input_tensors_.emplace(&mul->input_a(), op.inputs()->Get(0));
    input_tensors_.emplace(&max->input_a(), op.inputs()->Get(0));
    output_tensors_.emplace(op.outputs()->Get(0), &max->output());
}
