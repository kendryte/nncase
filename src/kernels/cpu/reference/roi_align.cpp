/* Copyright 2019-2021 Canaan Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <nncase/kernels/cpu/reference/tensor_compute.h>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::cpu;
using namespace nncase::kernels::cpu::reference;

namespace
{
template <typename T>
struct PreCalc
{
    int64_t pos1;
    int64_t pos2;
    int64_t pos3;
    int64_t pos4;
    T w1;
    T w2;
    T w3;
    T w4;
};

template <typename T>
static void PreCalcForBilinearInterpolate(const int64_t height, const int64_t width, const int64_t pooled_height,
    const int64_t pooled_width, const int64_t iy_upper, const int64_t ix_upper,
    T roi_start_h, T roi_start_w, T bin_size_h, T bin_size_w, int64_t roi_bin_grid_h,
    int64_t roi_bin_grid_w, std::vector<PreCalc<T>> &pre_calc)
{
    int64_t pre_calc_index = 0;
    for (int64_t ph = 0; ph < pooled_height; ph++)
    {
        for (int64_t pw = 0; pw < pooled_width; pw++)
        {
            for (int64_t iy = 0; iy < iy_upper; iy++)
            {
                const T yy = roi_start_h + ph * bin_size_h + static_cast<T>(iy + .5f) * bin_size_h / static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
                for (int64_t ix = 0; ix < ix_upper; ix++)
                {
                    const T xx = roi_start_w + pw * bin_size_w + static_cast<T>(ix + .5f) * bin_size_w / static_cast<T>(roi_bin_grid_w);

                    T x = xx;
                    T y = yy;
                    // deal with: inverse elements are out of feature map boundary
                    if (y < -1.0 || y > height || x < -1.0 || x > width)
                    {
                        auto &pc = pre_calc[pre_calc_index];
                        pc.pos1 = 0;
                        pc.pos2 = 0;
                        pc.pos3 = 0;
                        pc.pos4 = 0;
                        pc.w1 = 0;
                        pc.w2 = 0;
                        pc.w3 = 0;
                        pc.w4 = 0;
                        pre_calc_index += 1;
                        continue;
                    }

                    if (y <= 0)
                    {
                        y = 0;
                    }
                    if (x <= 0)
                    {
                        x = 0;
                    }

                    auto y_low = static_cast<int64_t>(y);
                    auto x_low = static_cast<int64_t>(x);
                    int64_t y_high;
                    int64_t x_high;

                    if (y_low >= height - 1)
                    {
                        y_high = y_low = height - 1;
                        y = (T)y_low;
                    }
                    else
                    {
                        y_high = y_low + 1;
                    }

                    if (x_low >= width - 1)
                    {
                        x_high = x_low = width - 1;
                        x = (T)x_low;
                    }
                    else
                    {
                        x_high = x_low + 1;
                    }

                    T ly = y - y_low;
                    T lx = x - x_low;
                    T hy = static_cast<T>(1.) - ly;
                    T hx = static_cast<T>(1.) - lx;
                    T w1 = hy * hx;
                    T w2 = hy * lx;
                    T w3 = ly * hx;
                    T w4 = ly * lx;

                    // save weights and indeces
                    PreCalc<T> pc;
                    pc.pos1 = y_low * width + x_low;
                    pc.pos2 = y_low * width + x_high;
                    pc.pos3 = y_high * width + x_low;
                    pc.pos4 = y_high * width + x_high;
                    pc.w1 = w1;
                    pc.w2 = w2;
                    pc.w3 = w3;
                    pc.w4 = w4;
                    pre_calc[pre_calc_index] = pc;

                    pre_calc_index += 1;
                }
            }
        }
    }
}
}

template result<void> reference::roi_align<float>(const float *input, const float *rois, int64_t *batch_indices, float *output,
    const runtime_shape_t &in_shape, const runtime_shape_t &output_shape, roi_align_mode_t mode, float spatial_scale,
    int64_t sampling_ratio) noexcept;

template <typename T>
result<void> reference::roi_align(const T *input, const T *rois, int64_t *batch_indices, T *output,
    const runtime_shape_t &in_shape, const runtime_shape_t &output_shape, roi_align_mode_t mode, float spatial_scale,
    int64_t sampling_ratio) noexcept
{
    int64_t height = in_shape[2];
    int64_t width = in_shape[3];

    size_t n_rois = output_shape[0];
    int64_t channels = output_shape[1];
    int64_t pooled_height = output_shape[2];
    int64_t pooled_width = output_shape[3];

    size_t num_roi_cols = 4;

    //100 is a random chosed value, need be tuned
    // double cost = static_cast<double>(channels * pooled_width * pooled_height * 100);

    for (size_t n = 0; n < n_rois; n++)
    {
        int64_t index_n = n * channels * pooled_width * pooled_height;

        const T *offset_rois = rois + n * num_roi_cols;
        const auto roi_batch_ind = batch_indices[n];

        // Do not using rounding; this implementation detail is critical
        T roi_start_w = offset_rois[0] * spatial_scale;
        T roi_start_h = offset_rois[1] * spatial_scale;
        T roi_end_w = offset_rois[2] * spatial_scale;
        T roi_end_h = offset_rois[3] * spatial_scale;

        // Force malformed ROIs to be 1x1
        T roi_width = std::max(roi_end_w - roi_start_w, (T)1.);
        T roi_height = std::max(roi_end_h - roi_start_h, (T)1.);
        T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
        T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

        // We use roi_bin_grid to sample the grid and mimic integral
        int64_t roi_bin_grid_h = (sampling_ratio > 0) ? sampling_ratio : static_cast<int64_t>(std::ceil(roi_height / pooled_height)); // e.g., = 2
        int64_t roi_bin_grid_w = (sampling_ratio > 0) ? sampling_ratio : static_cast<int64_t>(std::ceil(roi_width / pooled_width));

        // We do average (integral) pooling inside a bin
        const int64_t count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

        // we want to precalculate indices and weights shared by all channels,
        // this is the key point of optimization
        std::vector<PreCalc<T>> pre_calc(roi_bin_grid_h * roi_bin_grid_w * pooled_width * pooled_height);
        PreCalcForBilinearInterpolate(height, width, pooled_height, pooled_width, roi_bin_grid_h, roi_bin_grid_w,
            roi_start_h, roi_start_w, bin_size_h, bin_size_w, roi_bin_grid_h,
            roi_bin_grid_w, pre_calc);

        for (int64_t c = 0; c < channels; c++)
        {
            int64_t index_n_c = index_n + c * pooled_width * pooled_height;
            const T *offset_input = input + static_cast<int64_t>((roi_batch_ind * channels + c) * height * width);
            int64_t pre_calc_index = 0;

            for (int64_t ph = 0; ph < pooled_height; ph++)
            {
                for (int64_t pw = 0; pw < pooled_width; pw++)
                {
                    int64_t index = index_n_c + ph * pooled_width + pw;

                    T output_val = 0.;
                    if (mode == roi_align_mode_t::roi_align_avg)
                    { // avg pooling
                        for (int64_t iy = 0; iy < roi_bin_grid_h; iy++)
                        {
                            for (int64_t ix = 0; ix < roi_bin_grid_w; ix++)
                            {
                                const auto &pc = pre_calc[pre_calc_index];
                                output_val += pc.w1 * offset_input[pc.pos1] + pc.w2 * offset_input[pc.pos2] + pc.w3 * offset_input[pc.pos3] + pc.w4 * offset_input[pc.pos4];

                                pre_calc_index += 1;
                            }
                        }
                        output_val /= count;
                    }
                    else
                    { // max pooling
                        bool max_flag = false;
                        for (int64_t iy = 0; iy < roi_bin_grid_h; iy++)
                        {
                            for (int64_t ix = 0; ix < roi_bin_grid_w; ix++)
                            {
                                const auto &pc = pre_calc[pre_calc_index];
                                T val = std::max(
                                    std::max(std::max(pc.w1 * offset_input[pc.pos1], pc.w2 * offset_input[pc.pos2]),
                                        pc.w3 * offset_input[pc.pos3]),
                                    pc.w4 * offset_input[pc.pos4]);
                                if (!max_flag)
                                {
                                    output_val = val;
                                    max_flag = true;
                                }
                                else
                                {
                                    output_val = std::max(output_val, val);
                                }

                                pre_calc_index += 1;
                            }
                        }
                    }

                    output[index] = output_val;
                } // for pw
            } // for ph
        } // for c
    } // for n

    return ok();
}
