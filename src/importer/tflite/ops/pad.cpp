#include "../tflite_importer.h"
#include <ir/ops/pad.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

DEFINE_TFLITE_LOWER(PAD)
{
    auto &input = get_tensor(op.inputs(), 0);
    auto &paddings = load_tensor<int32_t, 2>(get_tensor(op.inputs(), 1));
    auto &options = *op.builtin_options_as_PadOptions();

    xt::svector<padding> new_paddings;
    for (size_t i = 0; i < paddings.shape()[0]; i++)
        new_paddings.push_back(padding { paddings(i, 0), paddings(i, 1) });

    auto node = graph_.emplace<pad>(dt_float32, get_shape(*input.shape()), new_paddings, 0.f);

    input_tensors_.emplace(&node->input(), op.inputs()->Get(0));
    output_tensors_.emplace(op.outputs()->Get(0), &node->output());
}
