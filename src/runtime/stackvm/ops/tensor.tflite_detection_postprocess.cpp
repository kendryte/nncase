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
#include "../runtime_function.h"
#include <nncase/kernels/tensor_compute.h>
#include <nncase/runtime/interpreter.h>
#include <nncase/runtime/runtime_op_utility.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;

result<void> stackvm_runtime_function::visit(const tensor_tflite_detection_postprocess_op_t &op) noexcept
{
    try_var(output_num_detections, pop_addr());
    try_var(output_scores, pop_addr());
    try_var(output_classes, pop_addr());
    try_var(output_locations, pop_addr());
    try_var(anchor, pop_addr());
    try_var(score, pop_addr());
    try_var(box, pop_addr());

    try_var(box_shape, module().shape_reg(op.box_shape_src));
    try_var(score_shape, module().shape_reg(op.score_shape_src));
    try_var(anchor_shape, module().shape_reg(op.anchor_shape_src));

    return kernels::tflite_detection_postprocess(reinterpret_cast<const float *>(box), reinterpret_cast<const float *>(score),
        reinterpret_cast<const float *>(anchor), reinterpret_cast<float *>(output_locations),
        reinterpret_cast<float *>(output_classes), reinterpret_cast<float *>(output_scores),
        reinterpret_cast<float *>(output_num_detections), box_shape, score_shape, anchor_shape, op.max_detections, op.max_classes_per_detection, op.detections_per_class,
        op.use_regular_non_max_suppression, op.nms_score_threshold, op.nms_iou_threshold,
        op.num_classes, op.y_scale, op.x_scale, op.h_scale, op.w_scale);
}
