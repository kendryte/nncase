#include <ir/ops/transpose.h>
#include <ir/op_utils.h>
#include <xtensor/xarray.hpp>

using namespace nncase;
using namespace nncase::ir;

transpose::transpose(datatype_t type, shape_t input_shape, axis_t perm)
    : perm_(std::move(perm))
{
    add_input("input", type, input_shape);
    add_output("output", type, get_transposed_shape(input_shape, perm));
}
