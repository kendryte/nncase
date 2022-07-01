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
#include <chrono>
#include <iostream>
#include <nncase/kernels/cpu/reference/tensor_compute.h>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>
using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::cpu;
using namespace nncase::kernels::cpu::reference;

template result<void> reference::tflite_detection_postprocess<float>(const float *boxes, const float *scores, const float *anchors, float *output_locations, float *output_classes, float *output_scores, float *output_num_detections,
    const runtime_shape_t &boxes_shape, const runtime_shape_t &scores_shape, const runtime_shape_t &anchors_shape,
    const int32_t max_detections, const int32_t max_classes_per_detection, const int32_t detections_per_class,
    const bool use_regular_non_max_suppression, const float nms_score_threshold, const float nms_iou_threshold,
    const int32_t num_classes, const float y_scale, const float x_scale, const float h_scale, const float w_scale) noexcept;

template <typename T>
result<void> reference::tflite_detection_postprocess(const T *boxes, const T *scores, const T *anchors, T *output_locations, T *output_classes, T *output_scores, T *output_num_detections,
    const runtime_shape_t &boxes_shape, const runtime_shape_t &scores_shape, const runtime_shape_t &anchors_shape,
    const int32_t max_detections, const int32_t max_classes_per_detection, const int32_t detections_per_class,
    const bool use_regular_non_max_suppression, const float nms_score_threshold, const float nms_iou_threshold,
    const int32_t num_classes, const float y_scale, const float x_scale, const float h_scale, const float w_scale) noexcept
{
    struct CenterSizeEncoding
    {
        float y;
        float x;
        float h;
        float w;
    };
    struct BoxCornerEncoding
    {
        float ymin;
        float xmin;
        float ymax;
        float xmax;
    };
    struct BoxInfo
    {
        int index;
        float score;
    };

    auto compute_iou = [&](const std::vector<BoxCornerEncoding> &box, const int &i, const int &j) {
        auto &box_i = box[i];
        auto &box_j = box[j];
        const float area_i = (box_i.ymax - box_i.ymin) * (box_i.xmax - box_i.xmin);
        const float area_j = (box_j.ymax - box_j.ymin) * (box_j.xmax - box_j.xmin);
        if (area_i <= 0 || area_j <= 0)
            return 0.f;
        const float intersection_y_min = std::max<float>(box_i.ymin, box_j.ymin);
        const float intersection_x_min = std::max<float>(box_i.xmin, box_j.xmin);
        const float intersection_y_max = std::min<float>(box_i.ymax, box_j.ymax);
        const float intersection_x_max = std::min<float>(box_i.xmax, box_j.xmax);
        const float intersection_area = std::max<float>(intersection_y_max - intersection_y_min, 0.0) * std::max<float>(intersection_x_max - intersection_x_min, 0.0);
        return intersection_area / (area_i + area_j - intersection_area);
    };

    const auto num_boxes = (int)anchors_shape[0];
    const auto num_classes_with_background = (int)scores_shape[2]; // num_classes + background
    const auto num_detections_per_class = std::min(detections_per_class, max_detections);
    int label_offset = num_classes_with_background - num_classes;
    // DecodeCenterSizeBoxes： get decoded_boxes
    std::vector<BoxCornerEncoding> decoded_boxes(boxes_shape[1]);
    {
        CenterSizeEncoding box_center_size;
        CenterSizeEncoding scale_values { y_scale, x_scale, h_scale, w_scale };
        CenterSizeEncoding anchor;

        for (int index = 0; index < num_boxes; index++)
        {
            const auto box_encoding_index = index * boxes_shape[2];
            box_center_size = *reinterpret_cast<const CenterSizeEncoding *>(boxes + box_encoding_index);
            anchor = *reinterpret_cast<const CenterSizeEncoding *>(anchors + box_encoding_index);

            auto y_center = static_cast<float>(static_cast<double>(box_center_size.y) / static_cast<double>(scale_values.y) * static_cast<double>(anchor.h) + static_cast<double>(anchor.y));
            auto x_center = static_cast<float>(static_cast<double>(box_center_size.x) / static_cast<double>(scale_values.x) * static_cast<double>(anchor.w) + static_cast<double>(anchor.x));
            auto half_h = static_cast<float>(0.5 * (std::exp(static_cast<double>(box_center_size.h) / static_cast<double>(scale_values.h))) * static_cast<double>(anchor.h));
            auto half_w = static_cast<float>(0.5 * (std::exp(static_cast<double>(box_center_size.w) / static_cast<double>(scale_values.w))) * static_cast<double>(anchor.w));
            decoded_boxes[index].ymin = y_center - half_h;
            decoded_boxes[index].xmin = x_center - half_w;
            decoded_boxes[index].ymax = y_center + half_h;
            decoded_boxes[index].xmax = x_center + half_w;
        }
    }
    // NMS MultiClass
    {
        if (use_regular_non_max_suppression)
        {
            // NMS Regular
            int sorted_indices_size = 0;
            std::vector<BoxInfo> box_info_after_regular_nms(max_detections + num_detections_per_class);
            std::vector<int> num_selected(num_classes);

            // compute nms
            std::vector<float> class_scores(num_boxes);
            std::vector<int> selected;
            selected.reserve(num_detections_per_class);

            for (auto col = 0; col < num_classes - 1; col++)
            {
                const float *scores_base = scores + col + label_offset;
                for (int row = 0; row < num_boxes; row++)
                {
                    // Get scores of boxes corresponding to all anchors for single class
                    class_scores[row] = *scores_base;
                    scores_base += num_classes_with_background;
                }
                // Perform non-maximal suppression on single class
                selected.clear();

                // NMS SingleClass
                {
                    std::vector<int> keep_indices;
                    std::vector<float> keep_scores;
                    // select detection box score above score threshold
                    {
                        for (size_t i = 0; i < class_scores.size(); i++)
                        {
                            if (class_scores[i] >= nms_score_threshold)
                            {
                                keep_scores.emplace_back(class_scores[i]);
                                keep_indices.emplace_back(i);
                            }
                        }
                    }

                    int num_scores_kept = (int)keep_scores.size();
                    std::vector<int> sorted_indices;
                    sorted_indices.resize(num_scores_kept);
                    // DecreasingArgSort
                    {
                        std::iota(sorted_indices.begin(), sorted_indices.begin() + num_scores_kept, 0);
                        std::stable_sort(
                            sorted_indices.begin(), sorted_indices.begin() + num_scores_kept,
                            [&keep_scores](const int i, const int j) { return keep_scores[i] > keep_scores[j]; });
                    }

                    const int output_size = std::min(num_scores_kept, max_detections);
                    selected.clear();
                    int num_active_candidate = num_scores_kept;
                    std::vector<uint8_t> active_box_candidate(num_scores_kept, 1);
                    for (int i = 0; i < num_scores_kept; ++i)
                    {
                        if (num_active_candidate == 0 || (int)selected.size() >= output_size)
                            break;
                        if (active_box_candidate[i] == 1)
                        {
                            selected.push_back(keep_indices[sorted_indices[i]]);
                            active_box_candidate[i] = 0;
                            num_active_candidate--;
                        }
                        else
                        {
                            continue;
                        }
                        for (int j = i + 1; j < num_scores_kept; ++j)
                        {
                            if (active_box_candidate[j] == 1)
                            {

                                float iou = compute_iou(
                                    decoded_boxes, keep_indices[sorted_indices[i]],
                                    keep_indices[sorted_indices[j]]);

                                if (iou > nms_iou_threshold)
                                {
                                    active_box_candidate[j] = 0;
                                    num_active_candidate--;
                                }
                            }
                        }
                    }
                }
                // end NMS SingleClass

                if (selected.empty())
                {
                    continue;
                }
                for (size_t i = 0; i < selected.size(); ++i)
                {
                    box_info_after_regular_nms[sorted_indices_size + i].score = class_scores[selected[i]];
                    box_info_after_regular_nms[sorted_indices_size + i].index = (selected[i] * num_classes_with_background + col + label_offset);
                }

                // In-place merge the original boxes and new selected boxes which are both
                // sorted by scores.
                std::inplace_merge(box_info_after_regular_nms.begin(), box_info_after_regular_nms.begin() + sorted_indices_size,
                    box_info_after_regular_nms.begin() + sorted_indices_size + selected.size(),
                    [](const BoxInfo &a, const BoxInfo &b) { return a.score >= b.score; });

                sorted_indices_size = std::min(sorted_indices_size + static_cast<int>(selected.size()), max_detections);
            }
            // end compute nms result

            // Allocate output tensors
            for (int output_box_index = 0; output_box_index < max_detections; output_box_index++)
            {
                if (output_box_index < sorted_indices_size)
                {
                    const int anchor_index = floor(
                        box_info_after_regular_nms[output_box_index].index / num_classes_with_background);
                    const int class_index = box_info_after_regular_nms[output_box_index].index - anchor_index * num_classes_with_background - label_offset;
                    const float selected_score = box_info_after_regular_nms[output_box_index].score;
                    // detection_boxes
                    reinterpret_cast<BoxCornerEncoding *>(output_locations)[output_box_index] = decoded_boxes[anchor_index];
                    // detection_classes
                    output_classes[output_box_index] = class_index;
                    // detection_scores
                    output_scores[output_box_index] = selected_score;
                }
                else
                {
                    // detection_boxes
                    reinterpret_cast<BoxCornerEncoding *>(output_locations)[output_box_index] = { 0.0f, 0.0f, 0.0f, 0.0f };
                    // detection_classes
                    output_classes[output_box_index] = 0.0f;
                    // detection_scores
                    output_scores[output_box_index] = 0.0f;
                }
            }
            output_num_detections[0] = sorted_indices_size;
            box_info_after_regular_nms.clear();
        }
        else
        {
            // Fast NMS

            const int max_categories_per_anchor = max_classes_per_detection;
            const int num_categories_per_anchor = std::min(max_categories_per_anchor, num_classes);

            std::vector<float> max_scores;
            max_scores.resize(num_boxes);
            std::vector<int> sorted_class_indices;
            sorted_class_indices.resize(num_boxes * num_categories_per_anchor);

            for (int row = 0; row < num_boxes; row++)
            {
                const float *box_scores = scores + row * num_classes_with_background + label_offset;
                int *class_indices = sorted_class_indices.data() + row * num_categories_per_anchor;

                // DecreasingPartialArgSort
                if (num_categories_per_anchor == 1)
                {
                    auto arg_max_vector = [&](const T *input_data, int size) {
                        T max_value = input_data[0];
                        int max_index = 0;
                        for (int i = 1; i < size; ++i)
                        {
                            // const T curr_value = input_data[i];
                            if (input_data[i] > max_value)
                            {
                                max_value = input_data[i];
                                max_index = i;
                            }
                        }
                        return max_index;
                    };
                    class_indices[0] = arg_max_vector(box_scores, num_classes);
                }
                else
                {
                    std::iota(class_indices, class_indices + num_classes, 0);
                    std::partial_sort(
                        class_indices, class_indices + num_categories_per_anchor, class_indices + num_classes,
                        [&box_scores](const int i, const int j) { return box_scores[i] > box_scores[j]; });
                }
                // end DecreasingPartialArgSort

                max_scores[row] = box_scores[class_indices[0]];
            }
            std::vector<int> selected;
            // NMS SingleClass
            {
                std::vector<int> keep_indices;
                std::vector<float> keep_scores;
                // select detection box score above score threshold
                {
                    for (size_t i = 0; i < max_scores.size(); i++)
                    {
                        if (max_scores[i] >= nms_score_threshold)
                        {
                            keep_scores.emplace_back(max_scores[i]);
                            keep_indices.emplace_back(i);
                        }
                    }
                }

                int num_scores_kept = (int)keep_scores.size();
                std::vector<int> sorted_indices;
                sorted_indices.resize(num_scores_kept);
                // DecreasingArgSort
                {
                    std::iota(sorted_indices.begin(), sorted_indices.begin() + num_scores_kept, 0);
                    std::stable_sort(
                        sorted_indices.begin(), sorted_indices.begin() + num_scores_kept,
                        [&keep_scores](const int i, const int j) { return keep_scores[i] > keep_scores[j]; });
                }
                const int output_size = std::min(num_scores_kept, max_detections);
                selected.clear();
                int num_active_candidate = num_scores_kept;
                std::vector<uint8_t> active_box_candidate(num_scores_kept, 1);
                for (int i = 0; i < num_scores_kept; ++i)
                {
                    if (num_active_candidate == 0 || (int)selected.size() >= output_size)
                        break;
                    if (active_box_candidate[i] == 1)
                    {
                        selected.push_back(keep_indices[sorted_indices[i]]);
                        active_box_candidate[i] = 0;
                        num_active_candidate--;
                    }
                    else
                    {
                        continue;
                    }
                    for (int j = i + 1; j < num_scores_kept; ++j)
                    {
                        if (active_box_candidate[j] == 1)
                        {

                            float iou = compute_iou(
                                decoded_boxes, keep_indices[sorted_indices[i]],
                                keep_indices[sorted_indices[j]]);
                            if (iou > nms_iou_threshold)
                            {
                                active_box_candidate[j] = 0;
                                num_active_candidate--;
                            }
                        }
                    }
                }
            }
            // end NMS SingleClass

            // Allocate output tensors
            int output_box_index = 0;
            for (const auto &selected_index : selected)
            {
                const float *box_scores = scores + selected_index * num_classes_with_background + label_offset;
                const int *class_indices = sorted_class_indices.data() + selected_index * num_categories_per_anchor;

                for (int col = 0; col < num_categories_per_anchor; ++col)
                {
                    int box_offset = max_categories_per_anchor * output_box_index + col;
                    // detection_boxes
                    reinterpret_cast<BoxCornerEncoding *>(output_locations)[box_offset] = decoded_boxes[selected_index];
                    // detection_classes
                    output_classes[box_offset] = class_indices[col];
                    // detection_scores
                    output_scores[box_offset] = box_scores[class_indices[col]];
                }
                output_box_index++;
            }
            output_num_detections[0] = output_box_index;
        }
    }

    return ok();
}
