#include "../tflite_importer.h"
#include <ir/ops/concat.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

DEFINE_TFLITE_LOWER(CONCATENATION)
{
    std::vector<shape_t> inputs_shape;
    auto &options = *op.builtin_options_as_ConcatenationOptions();

    for (auto &&in : *op.inputs())
    {
        auto &tensor = *subGraph_->tensors()->Get(in);
        inputs_shape.emplace_back(get_shape(*tensor.shape()));
    }

    auto con = graph_.emplace<concat>(dt_float32, inputs_shape, options.axis());

    for (size_t i = 0; i < op.inputs()->size(); i++)
        input_tensors_.emplace(&con->input_at(i), op.inputs()->Get(i));

    output_tensors_.emplace(op.outputs()->Get(0), &con->output());
}
