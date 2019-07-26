#pragma once
#include "../node.h"
#include <xtensor/xtensor.hpp>

namespace nncase
{
namespace ir
{
    class resize_image : public node
    {
    public:
        input_connector &input() { return input_at(0); }
        output_connector &output() { return output_at(0); }

        const std::array<int32_t, 2> &new_size() const noexcept { return new_size_; }
        image_resize_mode_t mode() const noexcept { return mode_; }
        bool align_corners() const noexcept { return align_corners_; }

        resize_image(datatype_t type, image_resize_mode_t mode, shape_t input_shape, std::array<int32_t, 2> new_size, bool align_corners);

        node_opcode opcode() const noexcept override { return op_resize_image; }

    private:
        std::array<int32_t, 2> new_size_;
        image_resize_mode_t mode_;
        bool align_corners_;
    };
}
}
