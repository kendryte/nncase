#include <ir/op_utils.h>
#include <ir/ops/resize_image.h>
#include <xtensor/xarray.hpp>

using namespace nncase;
using namespace nncase::ir;

resize_image::resize_image(datatype_t type, image_resize_mode_t mode, shape_t input_shape, std::array<int32_t, 2> new_size, bool align_corners)
    : new_size_(new_size), mode_(mode), align_corners_(align_corners)
{
    add_input("input", type, input_shape);
    add_output("output", type, get_resize_image_shape(input_shape, new_size));
}
