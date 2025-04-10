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
class attention_kv_cache_node;
using attention_kv_cache = object_t<attention_kv_cache_node>;

class NNCASE_API attention_kv_cache_node : public object_node {
    DEFINE_OBJECT_KIND(object_node, object_attention_kv_cache);

  public:
    attention_kv_cache_node(runtime::runtime_tensor kv_cache,
                            runtime::runtime_tensor seq_lens,
                            runtime::runtime_tensor context_lens,
                            runtime::runtime_tensor block_tables,
                            runtime::runtime_tensor slot_mapping);

    /** @brief Gets element type. */
    const datatype_t &dtype() const noexcept { return dtype_; }

    // tensor get_context_block_ids(int request_id, int layerid);
    // tensor get_block(int kind, object blockid);
    // tensor get_slots(tensor block, int startslot, int count);
    // tensor get_output_slot_ids(int kind, int layerId);
    // void update_output_slot(int kind, object slotid, tensor slot);

  private:
    datatype_t dtype_;
    tensor kv_cache_;
    tensor seq_lens_;
    tensor context_lens_;
    tensor block_tables_;
    tensor slot_mapping_;
};
} // namespace nncase
