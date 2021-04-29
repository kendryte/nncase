#include <llir/ops/split.h>
#include <hlir/op_utils.h>

nncase::llir::split::split(datatype_t dt, shape_t input_shape, int64_t axis, const std::vector<int64_t>& splits): _splits(splits), _axis(axis) {
    // Input type of the input shape
    add_input("input", dt, input_shape);
    // Add our multiple outputs, 1 per split
    for(int split = 0; split < splits.size(); split ++) {
        add_output("output_" + std::to_string(split), dt, hlir::get_split_shape(input_shape, axis, splits, split));
    }
}