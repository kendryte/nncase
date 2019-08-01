#include "../tflite_importer.h"
#include <ir/ops/binary.h>
#include <ir/ops/constant.h>
#include <ir/ops/reduce.h>
#include <ir/ops/unary.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

DEFINE_TFLITE_LOWER(L2_NORMALIZATION)
{
    auto &input = get_tensor(op.inputs(), 0);
    auto &options = *op.builtin_options_as_SoftmaxOptions();

    auto in_shape = get_shape(*input.shape());
    axis_t reduce_axis;
    if (in_shape.size() == 1)
    {
        reduce_axis.push_back(0);
    }
    else
    {
        for (size_t i = 1; i < in_shape.size(); i++)
            reduce_axis.push_back(i);
    }

    auto square = graph_.emplace<unary>(unary_square, in_shape);
    auto sum = graph_.emplace<reduce>(reduce_sum, square->output().shape(), reduce_axis, 0.f, true);
    auto epsilon = graph_.emplace<constant>(1e-10f);
    auto max = graph_.emplace<binary>(binary_max, sum->output().shape(), epsilon->output().shape(), value_range<float>::default());
    auto rsqrt = graph_.emplace<unary>(unary_rsqrt, max->output().shape());
    auto mul = graph_.emplace<binary>(binary_mul, in_shape, rsqrt->output().shape(), value_range<float>::default());

    sum->input().connect(square->output());
    max->input_a().connect(sum->output());
    max->input_b().connect(epsilon->output());
    rsqrt->input().connect(max->output());
    mul->input_b().connect(rsqrt->output());

    input_tensors_.emplace(&square->input(), op.inputs()->Get(0));
    input_tensors_.emplace(&mul->input_a(), op.inputs()->Get(0));
    output_tensors_.emplace(op.outputs()->Get(0), &mul->output());
}
