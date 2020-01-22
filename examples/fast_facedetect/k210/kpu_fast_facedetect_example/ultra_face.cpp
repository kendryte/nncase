/* Copyright 2018 Canaan Inc.
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
#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>
#include "ultra_face.h"

#define num_featuremap 4
#define hard_nms 1
#define blending_nms 2 /* mix nms was been proposaled in paper blaze face, aims to minimize the temporal jitter*/

#define clip(x, y) (x < 0 ? 0 : (x > y ? y : x))

typedef struct FaceInfo
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;

} FaceInfo;

class ultra_face
{
public:
    ultra_face(int input_width, int input_length,
               float score_threshold_, float iou_threshold_, int topk_ = -1)
    {
        score_threshold = score_threshold_;
        iou_threshold = iou_threshold_;
        in_w = input_width;
        in_h = input_length;
        w_h_list = {in_w, in_h};

        image_h = in_h;
        image_w = in_w;

        for(auto size : w_h_list)
        {
            std::vector<float> fm_item;
            for(float stride : strides)
            {
                fm_item.push_back(ceil(size / stride));
            }
            featuremap_size.push_back(fm_item);
        }

        for(auto size : w_h_list)
        {
            shrinkage_size.push_back(strides);
        }
        /* generate prior anchors */
        for(int index = 0; index < num_featuremap; index++)
        {
            float scale_w = in_w / shrinkage_size[0][index];
            float scale_h = in_h / shrinkage_size[1][index];
            for(int j = 0; j < featuremap_size[1][index]; j++)
            {
                for(int i = 0; i < featuremap_size[0][index]; i++)
                {
                    float x_center = (i + 0.5) / scale_w;
                    float y_center = (j + 0.5) / scale_h;

                    for(float k : min_boxes[index])
                    {
                        float w = k / in_w;
                        float h = k / in_h;
                        priors.push_back({clip(x_center, 1), clip(y_center, 1), clip(w, 1), clip(h, 1)});
                    }
                }
            }
        }
        /* generate prior anchors finished */

        num_anchors = priors.size();
    }

    void generateBBox(float *scores, float *boxes)
    {
        for(int i = 0; i < num_anchors; i++)
        {
            if(scores[i * 2 + 1] > score_threshold)
            {
                FaceInfo rects;
                float x_center = boxes[i * 4] * center_variance * priors[i][2] + priors[i][0];
                float y_center = boxes[i * 4 + 1] * center_variance * priors[i][3] + priors[i][1];
                float w = exp(boxes[i * 4 + 2] * size_variance) * priors[i][2];
                float h = exp(boxes[i * 4 + 3] * size_variance) * priors[i][3];

                rects.x1 = clip(x_center - w / 2.0, 1) * image_w;
                rects.y1 = clip(y_center - h / 2.0, 1) * image_h;
                rects.x2 = clip(x_center + w / 2.0, 1) * image_w;
                rects.y2 = clip(y_center + h / 2.0, 1) * image_h;
                rects.score = clip(scores[i * 2 + 1], 1);
                bbox_collection.push_back(rects);
            }
        }
    }

    std::vector<FaceInfo> &nms(int type)
    {
        auto &input = bbox_collection;
        std::sort(input.begin(), input.end(), [](const FaceInfo &a, const FaceInfo &b) { return a.score > b.score; });

        int box_num = input.size();

        std::vector<int> merged(box_num, 0);

        for(int i = 0; i < box_num; i++)
        {
            if(merged[i])
                continue;
            std::vector<FaceInfo> buf;

            buf.push_back(input[i]);
            merged[i] = 1;

            float h0 = input[i].y2 - input[i].y1 + 1;
            float w0 = input[i].x2 - input[i].x1 + 1;

            float area0 = h0 * w0;

            for(int j = i + 1; j < box_num; j++)
            {
                if(merged[j])
                    continue;

                float inner_x0 = input[i].x1 > input[j].x1 ? input[i].x1 : input[j].x1;
                float inner_y0 = input[i].y1 > input[j].y1 ? input[i].y1 : input[j].y1;

                float inner_x1 = input[i].x2 < input[j].x2 ? input[i].x2 : input[j].x2;
                float inner_y1 = input[i].y2 < input[j].y2 ? input[i].y2 : input[j].y2;

                float inner_h = inner_y1 - inner_y0 + 1;
                float inner_w = inner_x1 - inner_x0 + 1;

                if(inner_h <= 0 || inner_w <= 0)
                    continue;

                float inner_area = inner_h * inner_w;

                float h1 = input[j].y2 - input[j].y1 + 1;
                float w1 = input[j].x2 - input[j].x1 + 1;

                float area1 = h1 * w1;

                float score;

                score = inner_area / (area0 + area1 - inner_area);

                if(score > iou_threshold)
                {
                    merged[j] = 1;
                    buf.push_back(input[j]);
                }
            }

            switch(type)
            {
                case hard_nms:
                {
                    output.push_back(buf[0]);
                    break;
                }
                case blending_nms:
                {
                    float total = 0;
                    for(int i = 0; i < buf.size(); i++)
                    {
                        total += exp(buf[i].score);
                    }
                    FaceInfo rects{};
                    for(int i = 0; i < buf.size(); i++)
                    {
                        float rate = exp(buf[i].score) / total;
                        rects.x1 += buf[i].x1 * rate;
                        rects.y1 += buf[i].y1 * rate;
                        rects.x2 += buf[i].x2 * rate;
                        rects.y2 += buf[i].y2 * rate;
                        rects.score += buf[i].score * rate;
                    }
                    output.push_back(rects);
                    break;
                }
                default:
                {
                    printf("wrong type of nms.\n");
                    exit(-1);
                }
            }
        }

        return output;
    }

    void clear()
    {
        bbox_collection.clear();
        output.clear();
    }

private:
    int image_w;
    int image_h;

    int in_w;
    int in_h;
    int num_anchors;

    float score_threshold;
    float iou_threshold;

    const float mean_vals[3] = {127, 127, 127};
    const float norm_vals[3] = {1.0 / 128, 1.0 / 128, 1.0 / 128};

    const float center_variance = 0.1;
    const float size_variance = 0.2;
    const std::vector<std::vector<float>> min_boxes = {
        {10.0f, 16.0f, 24.0f},
        {32.0f, 48.0f},
        {64.0f, 96.0f},
        {128.0f, 192.0f, 256.0f}};
    const std::vector<float> strides = {8.0, 16.0, 32.0, 64.0};
    std::vector<std::vector<float>> featuremap_size;
    std::vector<std::vector<float>> shrinkage_size;
    std::vector<int> w_h_list;

    std::vector<std::vector<float>> priors = {};
    std::vector<FaceInfo> bbox_collection, output;
};

static std::unique_ptr<ultra_face> g_detector;

extern "C" {
int ultra_face_init(int input_width, int input_height,
                    float score_threshold, float iou_threshold, int topk)
{
    g_detector = std::make_unique<ultra_face>(input_width, input_height, score_threshold, iou_threshold, topk);
    return 0;
}

int ultra_face_detect(float *scores, float *boxes, callback_draw_box callback)
{
    g_detector->clear();
    g_detector->generateBBox(scores, boxes);
    auto &output = g_detector->nms(blending_nms);
    printf("%d\n", (int)output.size());
    for (auto &face : output)
    {
        //printf("%f, %f, %f, %f, %f\n", face.x1, face.x2, face.y1, face.y2, face.score);
        callback((int)roundf(face.x1), (int)roundf(face.y1), (int)roundf(face.x2), (int)roundf(face.y2), face.score);
    }
    return (int) output.size();
}
}
