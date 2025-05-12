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
#include "attention_config.h"
#include "paged_attention_enums.h"
#include <nncase/object.h>
#include <ranges>
#include <vector>

namespace nncase::llm {
using paged_kvcache_axes_t = itlib::small_vector<paged_kvcache_dim_kind, 8>;

class paged_attention_config_node : public attention_config_node {
    DEFINE_OBJECT_KIND(attention_config_node, object_paged_attention_config);

  public:
    paged_attention_config_node(
        size_t num_layers, size_t num_kv_heads, size_t head_dim,
        typecode_t kv_type, size_t block_size,
        const std::array<paged_kvcache_dim_kind, 6> &cache_layout,
        const std::vector<paged_kvcache_dim_kind> &packed_axes,
        const dims_t &lanes,
        const std::vector<paged_kvcache_dim_kind> &sharding_axes,
        const std::vector<dims_t> &axis_policies) noexcept
        : attention_config_node(num_layers, num_kv_heads, head_dim, kv_type),
          block_size_(block_size),
          cache_layout_(cache_layout),
          packed_axes_(packed_axes.begin(), packed_axes.end()),
          lanes_(lanes),
          sharding_axes_(sharding_axes.begin(), sharding_axes.end()),
          axis_policies_(axis_policies.begin(), axis_policies.end()) {}

    size_t block_size() const noexcept { return block_size_; }

    void block_size(size_t block_size) noexcept { block_size_ = block_size; }

    const std::array<paged_kvcache_dim_kind, 6> &cache_layout() const noexcept {
        return cache_layout_;
    }

    void cache_layout(
        const std::array<paged_kvcache_dim_kind, 6> &cache_layout) noexcept {
        cache_layout_ = cache_layout;
    }

    const std::array<paged_kvcache_dim_kind, 2> block_layout() const noexcept {
        std::array<paged_kvcache_dim_kind, 2> block_layout;
        size_t j = 0;
        for (size_t i = 0; i < 6; i++) {
            auto dim = cache_layout_[i];
            if ((dim == paged_kvcache_dim_kind::head_dim) ||
                (dim == paged_kvcache_dim_kind::block_size)) {
                block_layout[j++] = dim;
            }
        }
        return block_layout;
    }

    const auto &packed_axes() const noexcept { return packed_axes_; }

    void packed_axes(
        const std::vector<paged_kvcache_dim_kind> &packed_axes) noexcept {
        packed_axes_.clear();
        packed_axes_.assign(packed_axes.begin(), packed_axes.end());
    }

    const dims_t &lanes() const noexcept { return lanes_; }

    void lanes(const std::vector<int> &lanes) noexcept {
        lanes_.clear();
        lanes_.assign(lanes.begin(), lanes.end());
    }

    const auto &sharding_axes() const noexcept { return sharding_axes_; }

    void sharding_axes(
        const std::vector<paged_kvcache_dim_kind> &sharding_axes) noexcept {
        sharding_axes_.clear();
        sharding_axes_.assign(sharding_axes.begin(), sharding_axes.end());
    }

    const auto &axis_policies() const noexcept { return axis_policies_; }

    void axis_policies(const std::vector<dims_t> &axis_policies) noexcept {
        axis_policies_.clear();
        axis_policies_.assign(axis_policies.begin(), axis_policies.end());
    }

    void axis_policies(int32_t i, const dims_t axis_policy) noexcept {
        axis_policies_[i] = axis_policy;
    }

  private:
    size_t block_size_;
    std::array<paged_kvcache_dim_kind, 6> cache_layout_;
    paged_kvcache_axes_t packed_axes_;
    dims_t lanes_;
    paged_kvcache_axes_t sharding_axes_;
    itlib::small_vector<dims_t, 8> axis_policies_;
};

using paged_attention_config = object_t<paged_attention_config_node>;
} // namespace nncase::llm
