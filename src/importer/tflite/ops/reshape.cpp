#include "../tflite_importer.h"
#include <ir/ops/reshape.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

DEFINE_TFLITE_LOWER(RESHAPE)
{
    auto &input = get_tensor(op.inputs(), 0);
    auto &options = *op.builtin_options_as_ReshapeOptions();

    auto node = graph_.emplace<reshape>(to_data_type(input.type()), get_shape(*input.shape()), get_shape(*options.new_shape()));

    input_tensors_.emplace(&node->input(), op.inputs()->Get(0));
    output_tensors_.emplace(op.outputs()->Get(0), &node->output());
}
