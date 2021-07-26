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
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    typedef void (*callback_draw_box)(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float);

    int yolox_init(int target_size_, float prob_threshold_,
        float nms_threshold_, size_t num_class_);

    int yolox_detect(const float *boxes, callback_draw_box callback);

#ifdef __cplusplus
}
#endif
