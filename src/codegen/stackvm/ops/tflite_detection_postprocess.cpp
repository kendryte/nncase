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
#include "../module_builder.h"

using namespace nncase;
using namespace nncase::codegen;
using namespace nncase::codegen::stackvm;
using namespace nncase::ir;

void stackvm_module_builder::emit(tflite_detection_postprocess &node, stackvm_op_builder &builder)
{
    auto &box = allocation(node.boxes());
    auto &score = allocation(node.scores());
    auto &anchor = allocation(node.anchors());
    auto &output_locations = allocation(node.output_locations());
    auto &output_classes = allocation(node.output_classes());
    auto &output_scores = allocation(node.output_scores());
    auto &output_num_detections = allocation(node.output_num_detections());

    builder.lea_buffer(box);
    builder.lea_buffer(score);
    builder.lea_buffer(anchor);
    builder.lea_buffer(output_locations);
    builder.lea_buffer(output_classes);
    builder.lea_buffer(output_scores);
    builder.lea_buffer(output_num_detections);

    builder.stshape(0, box.shape);
    builder.stshape(1, score.shape);
    builder.stshape(2, anchor.shape);

    builder.tensor_tflite_detection_postprocess_(0, 1, 2, node.max_detections(), node.max_classes_per_detection(), node.detections_per_class(),
        node.use_regular_non_max_suppression(), node.nms_score_threshold(), node.nms_iou_threshold(),
        node.num_classes(), node.y_scale(), node.x_scale(), node.h_scale(), node.w_scale());
}
