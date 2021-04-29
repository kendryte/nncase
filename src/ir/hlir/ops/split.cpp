#include <hlir/ops/split.h>
#include <hlir/op_utils.h>
#include <llir/ops/split.h>

nncase::hlir::split::split(datatype_t dt, shape_t input_shape, int64_t axis, const std::vector<int64_t>& splits): _axis(axis), _splits(splits) {
    // Input type of the input shape
    add_input("input", dt, input_shape);
    // Add our multiple outputs, 1 per split
    for(int split = 0; split < splits.size(); split ++) {
        add_output("output_" + std::to_string(split), dt, get_split_shape(input_shape, axis, splits, split));
    }
}

void nncase::hlir::split::compile(hlir_compile_context &context) {
    auto l_c = context.graph.emplace<llir::split>(input().type(), input().shape(), _axis, _splits);
    context.add_input(input(), l_c->input());
    for(int i = 0; i < outputs().size(); i ++) {
        context.add_output(output_at(i), l_c->output_at(i));
    }
}