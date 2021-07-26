/* Copyright 2021 Canaan Inc.
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
#include <array>
#include <cmath>
#include <memory>
#include <type_traits>
#include <vector>
#include "yolox.h"

#define TEST 1

template <typename T>
struct Rect
{
    T x;
    T y;
    T width;
    T height;
    T area()
    {
        return width * height;
    }
};

template <bool can_do>
struct intersection_area_impl;

template <>
struct intersection_area_impl<true>
{
    template <typename T>
    static inline T do_it(const Rect<T> &a, const Rect<T> &b)
    {
        T x0 = std::max(a.x, b.x);
        T y0 = std::max(a.y, b.y);

        T x1 = std::min(a.x + a.width, b.x + b.width);
        T y1 = std::min(a.y + a.height, b.y + b.height);
        T h = y1 - y0 + 1;
        T w = x1 - x0 + 1;
        if(h < 0 or w < 0)
        {
            return 0;
        }
        return h * w;
    }
};

template <typename T>
inline T intersection_area(const Rect<T> &a, const Rect<T> &b)
{
    return intersection_area_impl<std::is_arithmetic<T>::value>::do_it(a, b);
}

struct Object
{
    Rect<float> rect;
    int label;
    float prob;
};

static inline float intersection_area(const Object &a, const Object &b)
{
    return intersection_area(a.rect, b.rect);
}

struct GridAndStride
{
    size_t grid0;
    size_t grid1;
    size_t stride;
};

class yolox_decoder
{
private:
    const size_t target_size;
    const float prob_threshold;
    const float nms_threshold;
    std::vector<GridAndStride> grid_strides;
    std::vector<Object> objects;
    std::vector<size_t> picked;
    const size_t num_class;

public:
    yolox_decoder()
        : target_size(224), prob_threshold(.3f), nms_threshold(.65f), num_class(85)
    {
    }
    yolox_decoder(const size_t target_size_, const float prob_threshold_,
                  const float nms_threshold_, const size_t num_class_)
        : target_size(target_size_), prob_threshold(prob_threshold_), nms_threshold(nms_threshold_), num_class(num_class_)
    {
    }

    template <size_t N>
    void generate_grids_and_stride(std::array<size_t, N> &strides)
    {
        for(auto stride : strides)
        {
            size_t num_grid = target_size / stride;
            for(size_t g1 = 0; g1 < num_grid; g1++)
            {
                for(size_t g0 = 0; g0 < num_grid; g0++)
                {
                    grid_strides.push_back({g0, g1, stride});
                }
            }
        }
    }

    void clear_resluts()
    {
        objects.clear();
        picked.clear();
    }

    void generate_proposals(const float *feat_ptr)
    {

        for(size_t anchor_idx = 0; anchor_idx < grid_strides.size(); anchor_idx++)
        {
            const int grid0 = grid_strides[anchor_idx].grid0;
            const int grid1 = grid_strides[anchor_idx].grid1;
            const int stride = grid_strides[anchor_idx].stride;

            // yolox/models/yolo_head.py decode logic
            //  outputs[..., :2] = (outputs[..., :2] + grids) * strides
            //  outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
            float x_center = (feat_ptr[0] + grid0) * stride;
            float y_center = (feat_ptr[1] + grid1) * stride;
            float w = expf(feat_ptr[2]) * stride;
            float h = expf(feat_ptr[3]) * stride;
            float x0 = x_center - w * 0.5f;
            float y0 = y_center - h * 0.5f;

            float box_objectness = feat_ptr[4];
            for(int class_idx = 0; class_idx < num_class; class_idx++)
            {
                float box_cls_score = feat_ptr[5 + class_idx];
                float box_prob = box_objectness * box_cls_score;
                if(box_prob > prob_threshold)
                {
                    Object obj;
                    obj.rect.x = x0;
                    obj.rect.y = y0;
                    obj.rect.width = w;
                    obj.rect.height = h;
                    obj.label = class_idx;
                    obj.prob = box_prob;
                    objects.push_back(obj);
                }

            } // class loop
            feat_ptr += (num_class + 5);

        } // point anchor loop
    }

    const std::vector<size_t> &nms()
    {
        std::sort(objects.begin(), objects.end(), [](const Object &a, const Object &b) { return a.prob > b.prob; });

        const size_t n = objects.size();
        std::vector<float> areas(n, 0);

        for(size_t i = 0; i < n; i++)
        {
            areas[i] = objects[i].rect.area();
        }

        for(size_t i = 0; i < n; i++)
        {
            const Object &a = objects[i];

            int keep = 1;
            for(size_t j = 0; j < picked.size(); j++)
            {
                const Object &b = objects[picked[j]];
                // intersection over union
                float inter_area = intersection_area(a, b);
                if(inter_area == 0)
                {
                    continue;
                }
                float union_area = areas[i] + areas[picked[j]] - inter_area;
                // float IoU = inter_area / union_area
                if(inter_area / union_area > nms_threshold)
                {
                    keep = 0;
                    break;
                }
            }
            if(keep)
            {
                picked.push_back(i);
            }
        }

        return picked;
    }

    const std::vector<Object> &get_objects()
    {
        return objects;
    }
};

static std::unique_ptr<yolox_decoder> detector;

extern "C" {
int yolox_init(int target_size_, float prob_threshold_,
               float nms_threshold_, size_t num_class_)
{
    detector = std::make_unique<yolox_decoder>(target_size_, prob_threshold_, nms_threshold_, num_class_);
    std::array<size_t, 3> strides{8, 16, 32};
    detector->generate_grids_and_stride(strides);
    return 0;
}

int yolox_detect(const float *boxes, callback_draw_box callback)
{
    detector->clear_resluts();
    detector->generate_proposals(boxes);
    auto &picked = detector->nms();
    auto &objects = detector->get_objects();
    for(auto &i : picked)
    {
        auto &obj = objects[i];
        callback((int)roundf(obj.rect.x), (int)roundf(obj.rect.y), (int)roundf(obj.rect.x + obj.rect.width), (int)roundf(obj.rect.y + obj.rect.height), obj.label, obj.prob);
    }
    return (int)picked.size();
}
}