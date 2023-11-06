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
#include <xtensor/xtensor.hpp>

namespace nncase::ir
{
class NNCASE_API tflite_detection_postprocess : public node
{
public:
    DEFINE_NODE_OPCODE(op_tflite_detection_postprocess);

    input_connector &boxes() { return input_at(0); }
    input_connector &scores() { return input_at(1); }
    input_connector &anchors() { return input_at(2); }
    output_connector &output_locations() { return output_at(0); }
    output_connector &output_classes() { return output_at(1); }
    output_connector &output_scores() { return output_at(2); }
    output_connector &output_num_detections() { return output_at(3); }

    int32_t max_detections() const noexcept { return max_detections_; }
    int32_t max_classes_per_detection() const noexcept { return max_classes_per_detection_; }
    int32_t detections_per_class() const noexcept { return detections_per_class_; }
    bool use_regular_non_max_suppression() const noexcept { return use_regular_non_max_suppression_; }
    float nms_score_threshold() const noexcept { return nms_score_threshold_; }
    float nms_iou_threshold() const noexcept { return nms_iou_threshold_; };
    int32_t num_classes() const noexcept { return num_classes_; };
    float y_scale() const noexcept { return y_scale_; };
    float x_scale() const noexcept { return x_scale_; };
    float h_scale() const noexcept { return h_scale_; };
    float w_scale() const noexcept { return w_scale_; };

    tflite_detection_postprocess(
        shape_t boxes_shape, shape_t scores_shape, shape_t anchors_shape,
        shape_t output_shape_0, shape_t output_shape_1, shape_t output_shape_2, shape_t output_shape_3,
        int32_t max_detections,
        int32_t max_classes_per_detection,
        int32_t detections_per_class,
        bool use_regular_non_max_suppression,
        float nms_score_threshold,
        float nms_iou_threshold,
        int32_t num_classes,
        float y_scale, float x_scale, float h_scale, float w_scale);

protected:
    bool properties_equal(node &other) const override;

private:
    int32_t max_detections_;
    int32_t max_classes_per_detection_;
    int32_t detections_per_class_;
    bool use_regular_non_max_suppression_;
    float nms_score_threshold_;
    float nms_iou_threshold_;
    int32_t num_classes_;
    float y_scale_;
    float x_scale_;
    float h_scale_;
    float w_scale_;
};
}
