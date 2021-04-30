#include <hlir/ops/upsample.h>
#include <hlir/op_utils.h>
#include <llir/ops/upsample.h>

nncase::hlir::upsample::upsample(datatype_t dt, shape_t input_shape, const std::vector<float> &scales): _scales(scales) {
    // Add our 2 inputs; one for the actual node, and one for our scales initializer
    add_input("input_tensor", dt, input_shape);

    // One output of the new scaled size
    add_output("output", dt, get_scaled_shape(input_shape, scales));
}

void nncase::hlir::upsample::compile(hlir_compile_context &context) {
    auto l_c = context.graph.emplace<llir::upsample>(input_at(0).type(), input_at(0).shape(), _scales);
    context.add_input(input_at(0), l_c->input_at(0));
    context.add_output(output_at(0), l_c->output_at(0));
}