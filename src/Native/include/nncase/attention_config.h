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
#pragma once
#include "object.h"
#include "shape.h"
#include "tensor.h"
#include "value.h"
#include <nncase/runtime/buffer.h>
#include <nncase/runtime/datatypes.h>
#include <nncase/runtime/runtime_tensor.h>

namespace nncase {

class attention_config_node : public object_node {
    DEFINE_OBJECT_KIND(object_node, object_attention_config);

  public:
    attention_config_node(int num_layers, int num_kv_heads, int head_dim)
        : num_layers(num_layers),
          num_kv_heads(num_kv_heads),
          head_dim(head_dim) {}

    int num_layers;
    int num_kv_heads;
    int head_dim;
};

using attention_config = object_t<attention_config_node>;
} // namespace nncase