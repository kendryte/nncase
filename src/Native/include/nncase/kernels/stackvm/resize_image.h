/* Copyright 2019-2023 Canaan Inc.
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
#include <nncase/runtime/stackvm/opcode.h>

using namespace nncase::runtime::stackvm;

using get_coordinate_func_t = float (*)(float, float, float, float, float,
                                        float);
using get_nearest_pixel_func_t = int64_t (*)(float);

get_coordinate_func_t get_coordinate_from_resized(
    image_resize_transformation_mode_t coordinate_transform_mode);

get_nearest_pixel_func_t
get_nearest_pixel_from_origin(image_resize_nearest_mode_t nearest_mode);

inline get_coordinate_func_t get_coordinate_from_resized(
    image_resize_transformation_mode_t coordinate_transform_mode) {
    switch (coordinate_transform_mode) {
    case image_resize_transformation_mode_t::asymmetric:
        return [](float x_resized, float x_scale, float, float, float, float) {
            return x_resized * x_scale;
        };
    case image_resize_transformation_mode_t::pytorch_half_pixel:
        return [](float x_resized, float x_scale, float length_resized, float,
                  float, float) {
            return length_resized > 1 ? (x_resized + 0.5f) * x_scale - 0.5f
                                      : 0.0f;
        };
    case image_resize_transformation_mode_t::align_corners:
        return [](float x_resized, float, float length_resized,
                  float length_original, float, float) {
            return length_resized == 1 ? 0
                                       : x_resized * (length_original - 1) /
                                             (length_resized - 1);
        };
    case image_resize_transformation_mode_t::tfcrop_and_resize:
        return [](float x_resized, float, float length_resized,
                  float length_original, float roi_start, float roi_end) {
            auto orig =
                length_resized > 1
                    ? roi_start * (length_original - 1) +
                          (x_resized * (roi_end - roi_start) *
                           (length_original - 1)) /
                              (length_resized - 1)
                    : 0.5 * (roi_start + roi_end) * (length_original - 1);
            return static_cast<float>(orig);
        };
    default: // "image_resize_transformation_mode_t::half_pixel"
        return [](float x_resized, float x_scale, float, float, float, float) {
            return ((x_resized + 0.5f) * x_scale) - 0.5f;
        };
    }
}

inline get_nearest_pixel_func_t
get_nearest_pixel_from_origin(image_resize_nearest_mode_t nearest_mode) {
    switch (nearest_mode) {
    case image_resize_nearest_mode_t::round_prefer_ceil:
        return [](float x_original) {
            return static_cast<int64_t>(roundf(x_original));
        };
    case image_resize_nearest_mode_t::floor:
        return [](float x_original) {
            return static_cast<int64_t>(std::floor(x_original));
        };
    case image_resize_nearest_mode_t::ceil:
        return [](float x_original) {
            return static_cast<int64_t>(std::ceil(x_original));
        };
    default: // default is round_prefer_floor
        return [](float x_original) {
            // for half way cases prefer floor
            if (x_original == static_cast<int64_t>(x_original) + 0.5f) {
                return static_cast<int64_t>(std::floor(x_original));
            }
            return static_cast<int64_t>(roundf(x_original));
        };
    }
}