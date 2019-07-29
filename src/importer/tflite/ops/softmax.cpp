#include "../tflite_importer.h"
#include <ir/ops/binary.h>
#include <ir/ops/constant.h>
#include <ir/ops/reduce.h>
#include <ir/ops/unary.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

DEFINE_TFLITE_LOWER(SOFTMAX)
{
    auto &input = get_tensor(op.inputs(), 0);
    auto &options = *op.builtin_options_as_SoftmaxOptions();

    auto in_shape = get_shape(*input.shape());
    xt::svector<int32_t> reduce_axis;
    if (in_shape.size() == 1)
    {
        reduce_axis.push_back(0);
    }
    else
    {
        for (size_t i = 1; i < in_shape.size(); i++)
            reduce_axis.push_back(i);
    }

    auto max = graph_.emplace<reduce>(reduce_max, in_shape, xt::adapt(reduce_axis), std::numeric_limits<float>::lowest(), false);
    auto sub = graph_.emplace<binary>(binary_sub, in_shape, max->output().shape(), value_range<float>::default());
    auto beta = graph_.emplace<constant>(options.beta());
    auto mul = graph_.emplace<binary>(binary_mul, sub->output().shape(), beta->output().shape(), value_range<float>::default());
    auto exp = graph_.emplace<unary>(unary_exp, mul->output().shape());

    //input_tensors_.emplace(&sm->input(), op.inputs()->Get(0));
    //output_tensors_.emplace(op.outputs()->Get(0), &sm->output());
}
