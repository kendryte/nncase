#include "resize_util.h"
#include "runtime_utils.h"
#include <apply.h>
#include <cmath>
#ifdef __riscv_vector_
#include <riscv_vector.h>
#endif

using namespace nncase::runtime::xpu;

namespace kernels {
namespace {
template <class T>
void resize_bilinear_impl(const T *input, T *output,
                          gsl::span<const size_t> in_shape,
                          gsl::span<const size_t> in_strides,
                          gsl::span<const size_t> out_strides, int32_t out_h,
                          int32_t out_w, bool align_corners,
                          bool half_pixel_centers) noexcept {
    auto [height_scale, width_scale] =
        get_resize_scales(in_shape, out_h, out_w, align_corners);

    const float rounding_offset =
        std::numeric_limits<T>::is_integer ? .5f : .0f;
    dims_t in_index(4), out_index(4);

    auto get_input = [&](int32_t in_y, int32_t in_x) {
        in_index[2] = in_y;
        in_index[3] = in_x;
        return input[offset(in_strides, in_index)];
    };

    for (size_t batch = 0; batch < in_shape[0]; batch++) {
        in_index[0] = batch;
        out_index[0] = batch;
        for (size_t oc = 0; oc < in_shape[1]; oc++) {
            in_index[1] = oc;
            out_index[1] = oc;
            for (size_t oy = 0; oy < (size_t)out_h; oy++) {
                out_index[2] = oy;
                float in_y;
                int32_t in_y0, in_y1;
                set_resize_bilinear(oy, height_scale, half_pixel_centers,
                                    in_shape[2], in_y, in_y0, in_y1);

                for (size_t ox = 0; ox < (size_t)out_w; ox++) {
                    out_index[3] = ox;
                    float in_x;
                    int32_t in_x0, in_x1;
                    set_resize_bilinear(ox, width_scale, half_pixel_centers,
                                        in_shape[3], in_x, in_x0, in_x1);

                    auto v0 = get_input(in_y0, in_x0);
                    auto v1 = get_input(in_y1, in_x0);
                    auto v2 = get_input(in_y0, in_x1);
                    auto v3 = get_input(in_y1, in_x1);

                    auto a0 = (1 - (in_y - in_y0)) * (1 - (in_x - in_x0));
                    auto a1 = (in_y - in_y0) * (1 - (in_x - in_x0));
                    auto a2 = (1 - (in_y - in_y0)) * (in_x - in_x0);
                    auto a3 = (in_y - in_y0) * (in_x - in_x0);
                    output[offset(out_strides, out_index)] =
                        T(v0 * a0 + v1 * a1 + v2 * a2 + v3 * a3 +
                          rounding_offset);
                }
            }
        }
    }
    return;
}

template <class T>
void resize_nearest_neighbor_impl(
    const T *input, T *output, gsl::span<const size_t> in_shape,
    gsl::span<const size_t> in_strides, gsl::span<const size_t> out_strides,
    int32_t out_h, int32_t out_w, [[maybe_unused]] bool align_corners,
    [[maybe_unused]] bool half_pixel_centers,
    get_coordinate_func_t get_coordinate_func,
    get_nearest_pixel_func_t get_nearset_func) noexcept {
    auto [height_scale, width_scale] =
        get_resize_scales(in_shape, out_h, out_w, align_corners);

    dims_t in_index(4), out_index(4);
    for (size_t batch = 0; batch < in_shape[0]; batch++) {
        in_index[0] = batch;
        out_index[0] = batch;
        for (size_t oc = 0; oc < in_shape[1]; oc++) {
            in_index[1] = oc;
            out_index[1] = oc;
            for (size_t oy = 0; oy < (size_t)out_h; oy++) {
                auto iy = get_coordinate_func(oy, height_scale, out_h,
                                              in_shape[2], 0, 0);
                int64_t in_y = get_nearset_func(iy);
                if (in_y < 0)
                    in_y = 0;
                if (in_y >= in_shape[2])
                    in_y = in_shape[2] - 1;
                in_index[2] = in_y;
                out_index[2] = oy;

                for (size_t ox = 0; ox < (size_t)out_w; ox++) {
                    auto ix = get_coordinate_func(ox, width_scale, out_w,
                                                  in_shape[3], 0, 0);
                    int64_t in_x = get_nearset_func(ix);
                    if (in_x < 0)
                        in_x = 0;
                    if (in_x >= in_shape[3])
                        in_x = in_shape[3] - 1;
                    in_index[3] = in_x;
                    out_index[3] = ox;
                    output[offset(out_strides, out_index)] =
                        input[offset(in_strides, in_index)];
                }
            }
        }
    }
    return;
}
} // namespace

template <class T>
void resize_bilinear(const T *input, T *output, dims_t in_shape,
                     strides_t in_strides, strides_t out_strides, int32_t out_h,
                     int32_t out_w, bool align_corners,
                     bool half_pixel_centers) noexcept {
    resize_bilinear_impl(input, output, in_shape, in_strides, out_strides,
                         out_h, out_w, align_corners, half_pixel_centers);
}

template <class T>
void resize_neareast_neighbor(
    const T *input, T *output, dims_t in_shape, strides_t in_strides,
    strides_t out_strides, int32_t out_h, int32_t out_w, bool align_corners,
    bool half_pixel_centers, get_coordinate_func_t get_coordinate_func,
    get_nearest_pixel_func_t get_nearset_func) noexcept {
    resize_nearest_neighbor_impl(input, output, in_shape, in_strides,
                                 out_strides, out_h, out_w, align_corners,
                                 half_pixel_centers, get_coordinate_func,
                                 get_nearset_func);
}
} // namespace kernels
