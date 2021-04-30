#include <llir/ops/upsample.h>
#include <hlir/op_utils.h>

nncase::llir::upsample::upsample(datatype_t dt, shape_t input_shape, const std::vector<float> &scales): _scales(scales) {
    // Add our 2 inputs; one for the actual node, and one for our scales initializer
    add_input("input_tensor", dt, input_shape);

    // One output of the new scaled size
    add_output("output", dt, hlir::get_scaled_shape(input_shape, scales));
}