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
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/tflite_detection_postprocess.h>
#include <nncase/ir/visitor.h>
#include <nncase/transforms/neutral/fix_output_shape.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;

bool tflite_detection_postprocess_transform::on_try_match(node &node, transform_context &context)
{
    if (auto tdp = node_cast<tflite_detection_postprocess>(node))
    {
        if (tdp->output_locations().shape() == shape_t { 1, (size_t)tdp->max_detections(), 4 })
            return false;
        context.inputs.emplace_back(&tdp->boxes());
        context.inputs.emplace_back(&tdp->scores());
        context.inputs.emplace_back(&tdp->anchors());

        context.outputs.emplace_back(&tdp->output_locations());
        context.outputs.emplace_back(&tdp->output_classes());
        context.outputs.emplace_back(&tdp->output_scores());
        context.outputs.emplace_back(&tdp->output_num_detections());

        context.matched_nodes.emplace_back(tdp);
        return true;
    }

    return false;
}

void tflite_detection_postprocess_transform::process(transform_context &context)
{
    auto &box = *context.inputs[0]->connection();
    auto &score = *context.inputs[1]->connection();
    auto &anchor = *context.inputs[2]->connection();
    auto output_locations = context.outputs[0]->connections();
    auto output_classes = context.outputs[1]->connections();
    auto output_scores = context.outputs[2]->connections();
    auto output_num_detections = context.outputs[3]->connections();

    auto &old_tdp = static_cast<tflite_detection_postprocess &>(*context.matched_nodes[0]);
    shape_t new_output_shape_0 { 1, (size_t)old_tdp.max_detections(), 4 };
    shape_t new_output_shape_1 { 1, (size_t)old_tdp.max_detections() };
    shape_t new_output_shape_2 { 1, (size_t)old_tdp.max_detections() };
    shape_t new_output_shape_3 { 1 };

    context.graph.outputs();
    auto new_output_node_0 = context.graph.emplace<output_node>(output_locations[0]->type(), new_output_shape_0);
    auto new_output_node_1 = context.graph.emplace<output_node>(output_classes[0]->type(), new_output_shape_1);
    auto new_output_node_2 = context.graph.emplace<output_node>(output_scores[0]->type(), new_output_shape_2);
    auto new_output_node_3 = context.graph.emplace<output_node>(output_num_detections[0]->type(), new_output_shape_3);
    new_output_node_0->name("output_locations");
    new_output_node_1->name("output_classes");
    new_output_node_2->name("output_scores");
    new_output_node_3->name("output_num_detections");

    auto new_tdp = context.graph.emplace<tflite_detection_postprocess>(old_tdp.boxes().shape(), old_tdp.scores().shape(), old_tdp.anchors().shape(),
        new_output_shape_0, new_output_shape_1, new_output_shape_2, new_output_shape_3, old_tdp.max_detections(), old_tdp.max_classes_per_detection(),
        old_tdp.detections_per_class(), old_tdp.use_regular_non_max_suppression(), old_tdp.nms_score_threshold(), old_tdp.nms_iou_threshold(),
        old_tdp.num_classes(), old_tdp.y_scale(), old_tdp.x_scale(), old_tdp.h_scale(), old_tdp.w_scale());
    new_tdp->name(old_tdp.name());

    for (auto &i : context.graph.outputs())
    {
        i->input().clear_connection();
    }

    new_tdp->boxes().connect(box);
    new_tdp->scores().connect(score);
    new_tdp->anchors().connect(anchor);

    new_output_node_0->input().connect(new_tdp->output_locations());
    new_output_node_1->input().connect(new_tdp->output_classes());
    new_output_node_2->input().connect(new_tdp->output_scores());
    new_output_node_3->input().connect(new_tdp->output_num_detections());

    context.graph.dce();
}