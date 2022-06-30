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
#include <nncase/ir/ops/tflite_detection_postprocess.h>
#include <xtensor/xarray.hpp>

using namespace nncase;
using namespace nncase::ir;

tflite_detection_postprocess::tflite_detection_postprocess(
    shape_t boxes_shape, shape_t scores_shape, shape_t anchors_shape,
    shape_t output_shape_0, shape_t output_shape_1, shape_t output_shape_2, shape_t output_shape_3,
    int32_t max_detections,
    int32_t max_classes_per_detection,
    int32_t detections_per_class,
    bool use_regular_non_max_suppression,
    float nms_score_threshold,
    float nms_iou_threshold,
    int32_t num_classes,
    float y_scale,
    float x_scale,
    float h_scale,
    float w_scale)
    : max_detections_(max_detections), max_classes_per_detection_(max_classes_per_detection), detections_per_class_(detections_per_class), use_regular_non_max_suppression_(use_regular_non_max_suppression), nms_score_threshold_(nms_score_threshold), nms_iou_threshold_(nms_iou_threshold), num_classes_(num_classes), y_scale_(y_scale), x_scale_(x_scale), h_scale_(h_scale), w_scale_(w_scale)
{
    add_input("boxes", dt_float32, boxes_shape);
    add_input("scores", dt_float32, scores_shape);
    add_input("anchors", dt_float32, anchors_shape);
    add_output("output_locations", dt_float32, output_shape_0);
    add_output("output_classes", dt_float32, output_shape_1);
    add_output("output_scores", dt_float32, output_shape_2);
    add_output("output_num_detections", dt_float32, output_shape_3);
}

bool tflite_detection_postprocess::properties_equal(node &other) const
{
    auto &r = static_cast<tflite_detection_postprocess &>(other);
    return max_detections() == r.max_detections()
        && max_classes_per_detection() == r.max_classes_per_detection()
        && detections_per_class() == r.detections_per_class()
        && use_regular_non_max_suppression() == r.use_regular_non_max_suppression()
        && nms_score_threshold() == r.nms_score_threshold()
        && nms_iou_threshold() == r.nms_iou_threshold() && num_classes() == r.num_classes()
        && y_scale() == r.y_scale() && x_scale() == r.x_scale() && h_scale() == r.h_scale() && w_scale() == r.w_scale();
}
