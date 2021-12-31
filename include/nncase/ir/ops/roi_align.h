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
#pragma once
#include "../node.h"

namespace nncase::ir
{
class NNCASE_API roi_align : public node
{
public:
    DEFINE_NODE_OPCODE(op_roi_align);

    input_connector &input() { return input_at(0); }
    input_connector &rois() { return input_at(1); }
    input_connector &batch_indices() { return input_at(2); }

    output_connector &output() { return output_at(0); }

    roi_align_mode_t mode() const noexcept { return mode_; }
    const float &spatial_scale() const noexcept { return spatial_scale_; }
    const int64_t &sampling_ratio() const noexcept { return sampling_ratio_; }

    roi_align(datatype_t input_type, shape_t input_shape, shape_t rois, shape_t batch_indices, roi_align_mode_t mode,
        float spatial_scale, int64_t output_height, int64_t output_width, int64_t sampling_ratio);

protected:
    bool properties_equal(node &other) const override;

private:
    roi_align_mode_t mode_;
    float spatial_scale_;
    int64_t sampling_ratio_;
};
}
