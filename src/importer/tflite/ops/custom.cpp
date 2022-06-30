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
#include "../tflite_importer.h"
#include <flatbuffers/flexbuffers.h>
#include <nncase/ir/ops/random_uniform.h>
#include <nncase/ir/ops/tflite_detection_postprocess.h>

using namespace nncase;
using namespace nncase::importer;
using namespace nncase::ir;

DEFINE_TFLITE_LOWER(CUSTOM)
{
    auto opcode = model_->operator_codes()->Get(op.opcode_index());
    auto custom_code = opcode->custom_code()->str();
    std::cout << "custom_code = " << custom_code << std::endl;
    if (custom_code == "FlexRandomUniform")
    {
        // auto custom_options = op.custom_options();
        // auto r = flexbuffers::GetRoot(custom_options->data(), custom_options->size());
        // std::cout << "Reference type: " << r.GetType() << std::endl;
        // auto v = flexbuffers::GetRoot(custom_options->data(), custom_options->size()).AsVector();
        // std::cout << "v.size = " << v.size() << std::endl;
        // for (size_t i = 0; i < v.size(); i++)
        // {
        //     std::cout << "i = " << i << ": Reference type: " << v[i].GetType() << std::endl;
        //     if (v[i].IsString())
        //         std::cout << v[i].AsString().str() << std::endl;
        // }
        auto &output = get_tensor(op.outputs(), 0);
        auto node = graph_.emplace<random_uniform>(to_data_type(output.type()), get_shape(output.shape()), 0.f, 1.f, time(nullptr));
        node->name(output.name()->string_view());
        link_output_tensor(op.outputs()->Get(0), &node->output());
    }
    else if (custom_code == "TFLite_Detection_PostProcess")
    {
        auto &input_decoded_boxes = get_tensor(op.inputs(), 0);
        auto &input_scores = get_tensor(op.inputs(), 1);
        auto &input_anchors = get_tensor(op.inputs(), 2);

        // get_shape(output_x.shape()): get error shape, ignore it in this step. fix it in independent transform
        auto &output_locations = get_tensor(op.outputs(), 0); //detection_boxes   (1, num_detected_boxes, 4)
        auto &output_classes = get_tensor(op.outputs(), 1); //detection_classes (1, num_detected_boxes)
        auto &output_scores = get_tensor(op.outputs(), 2); //detection_scores  (1, num_detected_boxes)
        auto &output_num_detections = get_tensor(op.outputs(), 3); //num_detections    (1)

        auto custom_options = op.custom_options();

        const auto &m = flexbuffers::GetRoot(custom_options->data(), custom_options->size()).AsMap();
        auto max_detections = m["max_detections"].AsInt32();
        auto max_classes_per_detection = m["max_classes_per_detection"].AsInt32();

        int32_t detections_per_class = 100;
        if (!m["detections_per_class"].IsNull())
            detections_per_class = m["detections_per_class"].AsInt32();

        bool use_regular_non_max_suppression = false;
        if (!m["use_regular_nms"].IsNull())
            use_regular_non_max_suppression = m["use_regular_nms"].AsBool();

        auto non_max_suppression_score_threshold = m["nms_score_threshold"].AsFloat();
        auto intersection_over_union_threshold = m["nms_iou_threshold"].AsFloat();
        auto num_classes = m["num_classes"].AsInt32();
        auto y = m["y_scale"].AsFloat();
        auto x = m["x_scale"].AsFloat();
        auto h = m["h_scale"].AsFloat();
        auto w = m["w_scale"].AsFloat();

        auto node = graph_.emplace<tflite_detection_postprocess>(get_shape(input_decoded_boxes.shape()), get_shape(input_scores.shape()), get_shape(input_anchors.shape()),
            get_shape(output_locations.shape()), get_shape(output_classes.shape()), get_shape(output_scores.shape()), get_shape(output_num_detections.shape()),
            max_detections, max_classes_per_detection, detections_per_class, use_regular_non_max_suppression, non_max_suppression_score_threshold,
            intersection_over_union_threshold, num_classes, y, x, h, w);

        link_input_tensor(&node->boxes(), op.inputs()->Get(0));
        link_input_tensor(&node->scores(), op.inputs()->Get(1));
        link_input_tensor(&node->anchors(), op.inputs()->Get(2));
        link_output_tensor(op.outputs()->Get(0), &node->output_locations());
        link_output_tensor(op.outputs()->Get(1), &node->output_classes());
        link_output_tensor(op.outputs()->Get(2), &node->output_scores());
        link_output_tensor(op.outputs()->Get(3), &node->output_num_detections());
    }
    else
    {
        throw std::runtime_error(std::string("Unsupported tflite CUSTOM code: ") + custom_code);
    }
}
