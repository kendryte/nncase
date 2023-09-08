#include "runtime_utils.h"

namespace kernels {

namespace {
template <typename T>
void matmul_unit_impl(const T *input_a, const T *input_b, T *output,
                      gsl::span<const size_t> in_a_shape,
                      gsl::span<const size_t> in_b_shape) noexcept {
    int32_t a_rows = static_cast<int32_t>(in_a_shape[0]);
    int32_t a_cols = static_cast<int32_t>(in_a_shape[1]);
    int32_t b_cols = static_cast<int32_t>(in_b_shape[1]);

    for (int32_t oy = 0; oy < a_rows; oy++) {
        T *values = (T *)runtime_util.malloc(sizeof(T) * b_cols);
        // runtime_util.memset(values, 0, sizeof(T) * b_cols);
        for (int32_t i = 0; i < a_cols; i++) {
            for (int32_t ox = 0; ox < b_cols; ox++) {
                const auto a = input_a[oy * a_cols + i];
                const auto b = input_b[i * b_cols + ox];
                values[ox] += a * b;
            }
        }
        runtime_util.memcpy(output + oy * b_cols, values, sizeof(T) * b_cols);
        runtime_util.free(values);
    }
}

template <typename T>
void contiguous_matmul_impl(const T *input_a, const T *input_b, T *output,
                            gsl::span<const size_t> in_a_shape,
                            gsl::span<const size_t> in_b_shape) noexcept {
    auto new_a_shape = to_nd(in_a_shape, 5);
    auto new_b_shape = to_nd(in_b_shape, 5);
    auto a_unit_size = new_a_shape[3] * new_a_shape[4];
    auto b_unit_size = new_b_shape[3] * new_b_shape[4];
    auto out_unit_size = new_a_shape[3] * new_b_shape[4];

    auto dim0 =
        new_a_shape[0] > new_b_shape[0] ? new_a_shape[0] : new_b_shape[0];
    auto dim1 =
        new_a_shape[1] > new_b_shape[1] ? new_a_shape[1] : new_b_shape[1];
    auto dim2 =
        new_a_shape[2] > new_b_shape[2] ? new_a_shape[2] : new_b_shape[2];
    auto ah_size = a_unit_size * new_a_shape[2];
    auto bh_size = b_unit_size * new_b_shape[2];
    auto oh_size = out_unit_size * dim2;

    auto ab_size = ah_size * new_a_shape[1];
    auto bb_size = bh_size * new_b_shape[1];
    auto ob_size = oh_size * dim1;

    for (size_t n = 0; n < dim0; ++n) {
        auto an = new_a_shape[0] == 1 ? 0 : n;
        auto bn = new_b_shape[0] == 1 ? 0 : n;
        for (size_t c = 0; c < dim1; ++c) {
            auto ac = new_a_shape[1] == 1 ? 0 : c;
            auto bc = new_b_shape[1] == 1 ? 0 : c;
            for (size_t h = 0; h < dim2; ++h) {
                auto ah = new_a_shape[2] == 1 ? 0 : h;
                auto bh = new_b_shape[2] == 1 ? 0 : h;
                matmul_unit_impl(
                    input_a + an * ab_size + ac * ah_size + ah * a_unit_size,
                    input_b + bn * bb_size + bc * bh_size + bh * b_unit_size,
                    output + n * ob_size + c * oh_size + h * out_unit_size,
                    std::array<size_t, 2>{new_a_shape[3], new_a_shape[4]},
                    std::array<size_t, 2>{new_b_shape[3], new_b_shape[4]});
            }
        }
    }
}

template <typename T>
void no_contiguous_matmul_impl(const T *input_a, const T *input_b, T *output,
                               gsl::span<const size_t> in_a_shape,
                               gsl::span<const size_t> in_a_stride,
                               gsl::span<const size_t> in_b_shape,
                               gsl::span<const size_t> in_b_stride,
                               gsl::span<const size_t> out_shape,
                               gsl::span<const size_t> out_stride) noexcept {
    auto [new_a_shape, new_a_stride] = to_nd(in_a_shape, in_a_stride, 5);
    auto [new_b_shape, new_b_stride] = to_nd(in_b_shape, in_b_stride, 5);
    auto [new_out_shape, new_out_stride] = to_nd(out_shape, out_stride, 5);

    for (size_t n = 0; n < new_out_shape[0]; ++n) {
        auto an = new_a_shape[0] == 1 ? 0 : n;
        auto bn = new_b_shape[0] == 1 ? 0 : n;
        for (size_t c = 0; c < new_out_shape[1]; ++c) {
            auto ac = new_a_shape[1] == 1 ? 0 : c;
            auto bc = new_b_shape[1] == 1 ? 0 : c;
            for (size_t h = 0; h < new_out_shape[2]; ++h) {
                auto ah = new_a_shape[2] == 1 ? 0 : h;
                auto bh = new_b_shape[2] == 1 ? 0 : h;

                const T *in_a_ptr = input_a + an * new_a_stride[0] +
                                    ac * new_a_stride[1] + ah * new_a_stride[2];
                const T *in_b_ptr = input_b + bn * new_b_stride[0] +
                                    bc * new_b_stride[1] + bh * new_b_stride[2];
                T *out_ptr = output + n * new_out_stride[0] +
                             c * new_out_stride[1] + h * new_out_stride[2];
                for (size_t m = 0; m < new_a_shape[3]; m++) {
                    T *values = (T*)runtime_util.malloc(new_b_shape[4] * sizeof(T));
                    for (size_t k = 0; k < new_a_shape[4]; k++) {
                        for (size_t n = 0; n < new_b_shape[4]; n++) {
                            values[n] += in_a_ptr[m * new_a_stride[3] +
                                                  k * new_a_stride[4]] *
                                         in_b_ptr[k * new_b_stride[3] +
                                                  n * new_b_stride[4]];
                        }
                    }
                    for (size_t n = 0; n < new_b_shape[4]; n++) {
                        out_ptr[m * new_out_stride[3] + n * new_out_stride[4]] =
                            values[n];
                    }
                    runtime_util.free(values);
                }
            }
        }
    }
}

} // namespace

template <class T>
void matmul(const T *input_a, const T *input_b, T *output, dims_t in_a_dims,
            strides_t in_a_strides, dims_t in_b_dims, strides_t in_b_strides,
            dims_t output_dims, strides_t output_strides

) {
    if (is_contiguous(in_a_dims, in_a_strides) &&
        is_contiguous(in_b_dims, in_b_strides) &&
        is_contiguous(output_dims, output_strides)) {
        return contiguous_matmul_impl(
            input_a, input_b, output,
            gsl::make_span(in_a_dims).template as_span<const size_t>(),
            gsl::make_span(in_b_dims).template as_span<const size_t>());
    } else {
        return no_contiguous_matmul_impl(
            input_a, input_b, output,
            gsl::make_span(in_a_dims).template as_span<const size_t>(),
            gsl::make_span(in_a_strides).template as_span<const size_t>(),
            gsl::make_span(in_b_dims).template as_span<const size_t>(),
            gsl::make_span(in_b_strides).template as_span<const size_t>(),
            gsl::make_span(output_dims).template as_span<const size_t>(),
            gsl::make_span(output_strides).template as_span<const size_t>());
    }
}
} // namespace kernels