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
#include <nncase/ir/op_utils.h>
#include <nncase/ir/ops/roi_align.h>
#include <xtensor/xarray.hpp>

using namespace nncase;
using namespace nncase::ir;

roi_align::roi_align(datatype_t input_type, shape_t input_shape, shape_t rois_shape, shape_t batch_indices_shape, roi_align_mode_t mode,
    float spatial_scale, int64_t output_height, int64_t output_width, int64_t sampling_ratio)
    : mode_(mode), spatial_scale_(spatial_scale), sampling_ratio_(sampling_ratio)
{
    shape_t out_shape { rois_shape[0], input_shape[1], static_cast<size_t>(output_height), static_cast<size_t>(output_width) };
    add_input("input", input_type, input_shape);
    add_input("rois", input_type, rois_shape);
    add_input("batch_indices", dt_int64, batch_indices_shape);
    add_output("output", input_type, out_shape);
}

bool roi_align::properties_equal(node &other) const
{
    auto &r = static_cast<roi_align &>(other);
    return mode() == r.mode() && spatial_scale() == r.spatial_scale() && sampling_ratio() == r.sampling_ratio();
}
