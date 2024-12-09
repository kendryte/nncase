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
#include "../../distributed.h"
#include "runtime.h"
#include <atomic>

namespace nncase::ntt::distributed {
template <> struct program_id_getter<topology::thread> {
    static size_t id() noexcept {
        return runtime::xpu_thread_context_t::current().tid;
    }
};

template <> struct program_id_getter<topology::block> {
    static size_t id() noexcept {
        return runtime::xpu_thread_context_t::current().bid;
    }
};

template <> struct program_id_getter<topology::die> {
    static size_t id() noexcept {
        return runtime::xpu_thread_context_t::current().did;
    }
};

template <> struct program_id_getter<topology::chip> {
    static size_t id() noexcept {
        return runtime::xpu_thread_context_t::current().cid;
    }
};

inline size_t tid() noexcept { return program_id<topology::thread>(); }
inline size_t bid() noexcept { return program_id<topology::block>(); }
inline size_t did() noexcept { return program_id<topology::die>(); }
inline size_t cid() noexcept { return program_id<topology::chip>(); }

inline constexpr size_t tdim() noexcept { return program_dim(topology::thread); }
inline constexpr size_t bdim() noexcept { return program_dim(topology::block); }
inline constexpr size_t ddim() noexcept { return program_dim(topology::die); }
inline constexpr size_t cdim() noexcept { return program_dim(topology::chip); }

namespace detail {
struct thread_barrier {
    std::atomic<int32_t> barrier[2] = { 0, 0 };
};

struct block_barrier {
    std::atomic<int32_t> barrier[2] = { 0, 0 };
};

struct die_barrier {
    std::atomic<int32_t> barrier[2] = { 0, 0 };
};

struct chip_barrier {
    std::atomic<int32_t> barrier[2] = { 0, 0 };
};

void arrive_and_wait(std::atomic<int32_t> vars[2], int32_t value = 0)
{
    vars[0].fetch_add(1, std::memory_order_seq_cst);
    while (vars[0].load() != value)
        ;

    vars[1].fetch_add(1, std::memory_order_seq_cst);
    while (vars[1].load() != value)
        ;

    vars[0].fetch_add(-1, std::memory_order_seq_cst);
    while (vars[0].load() != 0)
        ;

    vars[1].fetch_add(-1, std::memory_order_seq_cst);
    while (vars[1].load() != 0)
        ;    
}
} // namespace detail

template <> class topology_synchronizer<topology::thread> {
  private:
  public:
    static void synchronize() noexcept {
        detail::arrive_and_wait(barriers_[cid()][did()][bid()].barrier, tdim());
    }

  private:
    inline static detail::thread_barrier barriers_[cdim()][ddim()][bdim()];
};

template <> class topology_synchronizer<topology::block> {
  private:
  public:
    static void synchronize() noexcept {
        detail::arrive_and_wait(barriers_[cid()][did()].barrier,  bdim() * tdim());
    }

  private:
    inline static detail::block_barrier barriers_[cdim()][ddim()];
};

template <> class topology_synchronizer<topology::die> {
  private:
  public:
    static void synchronize() noexcept {
        detail::arrive_and_wait(barriers_[cid()].barrier,  ddim() * bdim() * tdim());
    }

  private:
    inline static detail::block_barrier barriers_[cdim()];
};

template <> class topology_synchronizer<topology::chip> {
  private:
  public:
    static void synchronize() noexcept {
        detail::arrive_and_wait(barrier_.barrier, cdim() * ddim() * bdim() * tdim());
    }

  private:
    inline static detail::chip_barrier barrier_;
};
} // namespace nncase::ntt::distributed
