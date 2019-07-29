#include "../tflite_importer.h"
#include <ir/ops/transpose.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

DEFINE_TFLITE_LOWER(TRANSPOSE)
{
    auto &input = get_tensor(op.inputs(), 0);
    auto &options = *op.builtin_options_as_TransposeOptions();
    auto perm = load_axis<int32_t>(get_tensor(op.inputs(), 1));

    auto node = graph_.emplace<transpose>(to_data_type(input.type()), get_shape(*input.shape()), perm);

    input_tensors_.emplace(&node->input(), op.inputs()->Get(0));
    output_tensors_.emplace(op.outputs()->Get(0), &node->output());
}
