/* Copyright 2019-2024 Canaan Inc.
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
#include <nncase/ntt/shape.h>
#define NNCASE_NTT_TOPOLOGY_DEFINED 1
namespace nncase::ntt::distributed {
constexpr auto topology_shape = ntt::fixed_shape_v<1, 1, 1>;
} // namespace nncase::ntt::distributed

#include "ntt_test.h"
#include "ortki_helper.h"
#include <cstdarg>
#include <cstddef>
#include <cstring>
#include <exception>
#include <gtest/gtest.h>
#include <iostream>
#include <nncase/ntt/arch/cpu/runtime.h>
#include <nncase/ntt/distributed.h>
#include <nncase/ntt/ntt.h>
#include <ortki/operators.h>
#include <thread>
#include <vector>

#ifdef WIN32
#include <Windows.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <mach/thread_policy.h>
#else
#include <pthread.h>
#endif

using namespace nncase;
using namespace nncase::ntt;
using namespace nncase::ntt::distributed;
using namespace nncase::ntt::distributed::shard_policy;
using namespace nncase::ntt::runtime;
using namespace ortki;

decltype(nncase::ntt::make_tensor<nncase::ntt::vector<uintptr_t, 2>>(
    nncase::ntt::distributed::topology_shape))
    nncase::ntt::distributed::detail::global_local_data_ptr =
        nncase::ntt::make_tensor<nncase::ntt::vector<uintptr_t, 2>>(
            nncase::ntt::distributed::topology_shape);

decltype(nncase::ntt::make_tensor<nncase::ntt::vector<uintptr_t, 2>>(
    nncase::ntt::distributed::topology_shape))
    nncase::ntt::distributed::detail::global_thread_local_rdata_ptr =
        nncase::ntt::make_tensor<nncase::ntt::vector<uintptr_t, 2>>(
            nncase::ntt::distributed::topology_shape);

decltype(nncase::ntt::make_tensor<nncase::ntt::vector<uintptr_t, 2>>(
    nncase::ntt::distributed::topology_shape))
    nncase::ntt::distributed::detail::global_block_local_rdata_ptr =
        nncase::ntt::make_tensor<nncase::ntt::vector<uintptr_t, 2>>(
            nncase::ntt::distributed::topology_shape);

namespace nncase::ntt::runtime {
size_t tdim;
size_t bdim;
size_t cdim;

#ifdef __APPLE__
pthread_key_t cpu_thread_context_key;
#else
thread_local cpu_thread_context_t cpu_thread_context;
#endif

void *thread_alloc(size_t bytes, size_t alignment) {
#ifdef WIN32
    return _aligned_malloc(bytes, alignment);
#else
    size_t mask = alignment - 1;
    size_t aligned_bytes = bytes + (-bytes & mask);
    auto ptr = aligned_alloc(alignment, aligned_bytes);
    if (!ptr) {
        std::terminate();
    }
    return ptr;
#endif
}

void thread_free(void *ptr) {
#ifdef WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}
} // namespace nncase::ntt::runtime

namespace tar {
uint8_t collective_pool_ptr[16 * 1024 * 1024] = {0};
} // namespace tar

using namespace nncase::ntt::runtime;

cpu_thread_context_t &cpu_thread_context_t::current() noexcept {
#ifndef __APPLE__
    return cpu_thread_context;
#else
    return *reinterpret_cast<cpu_thread_context_t *>(
        pthread_getspecific(cpu_thread_context_key));
#endif
}

void dump_mem(const std::string &info, const float *p, size_t num) {
    std::cout << info << std::endl;
    for (size_t i = 0; i < num; i++) {
        std::cout << "p[" << i << "] = " << p[i] << std::endl;
    }
}

// TEST(CpuTest, reshard_2D_same_sharding_spec_broadcast) {
//     // init
//     constexpr size_t M = 512;
//     constexpr size_t N = 1024;
//     using tensor_type = ntt::tensor<float, ntt::fixed_shape_v<M, N>>;
//     std::unique_ptr<tensor_type> ntt_input(new tensor_type);
//     NttTest::init_tensor(*ntt_input, -2.f, 2.f);
//     auto p_in = reinterpret_cast<float *>(ntt_input->elements().data());
// #ifdef __APPLE__
//     pthread_key_t cpu_thread_context_key_ = {};
//     pthread_key_create(&cpu_thread_context_key_, [](void *ptr) { delete
//     (nncase::ntt::runtime::cpu_thread_context_t *)ptr; });
//     cpu_thread_context_key = cpu_thread_context_key_;
// #endif

//     constexpr size_t cdims = ntt::distributed::cdim();
//     constexpr size_t bdims = ntt::distributed::bdim();
//     constexpr size_t tdims = ntt::distributed::tdim();
//     constexpr size_t num = cdims * bdims * tdims;

//     apply(global_local_data_ptr.shape(), [&](auto index) {
//         global_local_data_ptr(index)[0] =
//         (uintptr_t)(nncase::ntt::runtime::thread_alloc(M * N * sizeof(float),
//         8)); global_local_data_ptr(index)[1] =
//         global_local_data_ptr(index)[0] + M * N * sizeof(float);
//     });

//     std::vector<std::thread> threads;
//     for (size_t id = 0; id < num; id++) {
//         threads.emplace_back([cdims, bdims, tdims, id, M, N, p_in] {
//             size_t cid = id / (bdims * tdims);
//             size_t bid = id % (bdims * tdims) / tdims;
//             size_t tid = id % (bdims * tdims) % tdims;
// #ifdef __APPLE__
//             pthread_setspecific(cpu_thread_context_key,
//                                 new cpu_thread_context_t
// #else
//             cpu_thread_context_t::current() =
// #endif
//                                 {
//                                     .tid = tid,
//                                     .bid = bid,
//                                     .cid = cid,
//                                 }
// #ifdef __APPLE__
//             );
// #else
//             ;
// #endif

//             size_t cpu_id = id;
// #if WIN32
//             SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 <<
//             cpu_id);
// #elif defined(__APPLE__)
//             thread_affinity_policy_data_t policy = {(int)cpu_id};
//             thread_policy_set(pthread_mach_thread_np(pthread_self()),
//                               THREAD_AFFINITY_POLICY,
//                               (thread_policy_t)&policy,
//                               THREAD_AFFINITY_POLICY_COUNT);
// #else
//             cpu_set_t cpuset;
//             CPU_ZERO(&cpuset);
//             CPU_SET(cpu_id, &cpuset);
//             pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t),
//             &cpuset);
// #endif

//             dynamic_shape_t<ntt::distributed::topology_levels> mesh_index;
//             mesh_index[0] = cid;
//             mesh_index[1] = bid;
//             mesh_index[2] = tid;
//             auto local_data_lhs = reinterpret_cast<float
//             *>(global_local_data_ptr(mesh_index)[0]); auto local_data_rhs =
//             reinterpret_cast<float*>(nncase::ntt::runtime::thread_alloc(M * N
//             * sizeof(float), 8));

//             // shard(broadcast)
//             std::span<float, M * N> buffer(p_in, M * N);
//             tensor_view<float, fixed_shape_v<M, N>> tv(buffer);
//             sharded_tensor_view<float, fixed_shape_v<M, N>,
//             sharding<mesh<topology::thread, cdims, bdims, tdims>, B, I, I>>
//             stv_src(std::span<float, M * N>(local_data_lhs, M * N));
//             sharded_tensor_view<float, fixed_shape_v<M, N>,
//             sharding<mesh<topology::thread, cdims, bdims, tdims>, B, I, I>>
//             stv_dst(std::span<float, M * N>(local_data_rhs, M * N));

//             // shard
//             reshard(tv, stv_src);
//             EXPECT_TRUE(NttTest::compare_tensor(tv, stv_src.local()));
//             distributed::topology_synchronize();

//             // reshard
//             reshard(stv_src, stv_dst);
//             EXPECT_TRUE(NttTest::compare_tensor(tv, stv_dst.local()));

//             nncase::ntt::runtime::thread_free(local_data_rhs);
//         });
//     }

//     for (auto &t : threads)
//         t.join();

//     apply(global_local_data_ptr.shape(), [&](auto index) {
//         thread_free((void *)global_local_data_ptr(index)[0]);
//     });
// }

// TEST(CpuTest, reshard_2D_same_sharding_sepc_split) {
//     // init
//     constexpr size_t M = 512;
//     constexpr size_t N = 1024;
//     using tensor_type = ntt::tensor<float, ntt::fixed_shape_v<M, N>>;
//     std::unique_ptr<tensor_type> ntt_input(new tensor_type);
//     NttTest::init_tensor(*ntt_input, -2.f, 2.f);
//     auto p_in = reinterpret_cast<float *>(ntt_input->elements().data());

// #ifdef __APPLE__
//     pthread_key_t cpu_thread_context_key_ = {};
//     pthread_key_create(&cpu_thread_context_key_, [](void *ptr) { delete
//     (nncase::ntt::runtime::cpu_thread_context_t *)ptr; });
//     cpu_thread_context_key = cpu_thread_context_key_;
// #endif

//     constexpr size_t cdims = ntt::distributed::cdim();
//     constexpr size_t bdims = ntt::distributed::bdim();
//     constexpr size_t tdims = ntt::distributed::tdim();
//     constexpr size_t num = cdims * bdims * tdims;

//     apply(global_local_data_ptr.shape(), [&](auto index) {
//         global_local_data_ptr(index)[0] =
//         (uintptr_t)(nncase::ntt::runtime::thread_alloc(M * N * sizeof(float),
//         8)); global_local_data_ptr(index)[1] =
//         global_local_data_ptr(index)[0] + M * N * sizeof(float);
//     });

//     std::vector<std::thread> threads;
//     for (size_t id = 0; id < num; id++) {
//         threads.emplace_back([cdims, bdims, tdims, id, M, N, p_in] {
//             size_t cid = id / (bdims * tdims);
//             size_t bid = id % (bdims * tdims) / tdims;
//             size_t tid = id % (bdims * tdims) % tdims;
// #ifdef __APPLE__
//             pthread_setspecific(cpu_thread_context_key,
//                                 new cpu_thread_context_t
// #else
//             cpu_thread_context_t::current() =
// #endif
//                                 {
//                                     .tid = tid,
//                                     .bid = bid,
//                                     .cid = cid,
//                                 }
// #ifdef __APPLE__
//             );
// #else
//             ;
// #endif

//             size_t cpu_id = id;
// #if WIN32
//             SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 <<
//             cpu_id);
// #elif defined(__APPLE__)
//             thread_affinity_policy_data_t policy = {(int)cpu_id};
//             thread_policy_set(pthread_mach_thread_np(pthread_self()),
//                               THREAD_AFFINITY_POLICY,
//                               (thread_policy_t)&policy,
//                               THREAD_AFFINITY_POLICY_COUNT);
// #else
//             cpu_set_t cpuset;
//             CPU_ZERO(&cpuset);
//             CPU_SET(cpu_id, &cpuset);
//             pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t),
//             &cpuset);
// #endif

//             dynamic_shape_t<ntt::distributed::topology_levels> mesh_index;
//             mesh_index[0] = cid;
//             mesh_index[1] = bid;
//             mesh_index[2] = tid;
//             auto local_data_lhs = reinterpret_cast<float
//             *>(global_local_data_ptr(mesh_index)[0]); auto local_data_rhs =
//             reinterpret_cast<float *>(nncase::ntt::runtime::thread_alloc(M *
//             N * sizeof(float), 8));

//             // shard(slice)
//             constexpr size_t split_num = bdims * tdims;
//             tensor_view<float, fixed_shape_v<M, N>, fixed_strides<N, 1>>
//             tv(std::span<float, M * N>(p_in, M * N)); using sharding_type =
//             sharding<mesh<topology::thread, cdims, bdims, tdims>, B, S<1, 2>,
//             I>; sharded_tensor_view<float, fixed_shape_v<M, N>,
//             sharding_type> stv_src(std::span<float, M * N /
//             split_num>(local_data_lhs, M * N / split_num));
//             sharded_tensor_view<float, fixed_shape_v<M, N>, sharding_type>
//             stv_dst(std::span<float, M * N / split_num>(local_data_rhs, M * N
//             / split_num));

//             // shard
//             reshard(tv, stv_src);
//             distributed::topology_synchronize();

//             // reshard
//             reshard(stv_src, stv_dst);
//             EXPECT_TRUE(NttTest::compare_tensor(stv_src.local(),
//             stv_dst.local()));

//             nncase::ntt::runtime::thread_free(local_data_rhs);
//         });
//     }

//     for (auto &t : threads)
//         t.join();

//     apply(global_local_data_ptr.shape(), [&](auto index) {
//         thread_free((void *)global_local_data_ptr(index)[0]);
//     });
// }

// TEST(CpuTest, reshard_2D_different_sharding_spec_broadcast_split) {
//     // init
//     constexpr size_t M = 512;
//     constexpr size_t N = 1024;
//     using tensor_type = ntt::tensor<float, ntt::fixed_shape_v<M, N>>;
//     std::unique_ptr<tensor_type> ntt_input(new tensor_type);
//     std::unique_ptr<tensor_type> ntt_output(new tensor_type);

//     NttTest::init_tensor(*ntt_input, -2.f, 2.f);
//     NttTest::init_tensor(*ntt_output, -2.f, 2.f);

//     auto p_in = reinterpret_cast<float *>(ntt_input->elements().data());
//     auto p_out = reinterpret_cast<float *>(ntt_output->elements().data());

// #ifdef __APPLE__
//     pthread_key_t cpu_thread_context_key_ = {};
//     pthread_key_create(&cpu_thread_context_key_, [](void *ptr) { delete
//     (nncase::ntt::runtime::cpu_thread_context_t *)ptr; });
//     cpu_thread_context_key = cpu_thread_context_key_;
// #endif

//     constexpr size_t cdims = ntt::distributed::cdim();
//     constexpr size_t bdims = ntt::distributed::bdim();
//     constexpr size_t tdims = ntt::distributed::tdim();
//     constexpr size_t num = cdims * bdims * tdims;

//     apply(global_local_data_ptr.shape(), [&](auto index) {
//         global_local_data_ptr(index)[0] =
//         (uintptr_t)(nncase::ntt::runtime::thread_alloc(M * N * sizeof(float),
//         8)); global_local_data_ptr(index)[1] =
//         global_local_data_ptr(index)[0] + M * N * sizeof(float);
//     });

//     std::vector<std::thread> threads;
//     for (size_t id = 0; id < num; id++) {
//         threads.emplace_back([cdims, bdims, tdims, id, M, N, p_in, p_out] {
//             size_t cid = id / (bdims * tdims);
//             size_t bid = id % (bdims * tdims) / tdims;
//             size_t tid = id % (bdims * tdims) % tdims;
// #ifdef __APPLE__
//             pthread_setspecific(cpu_thread_context_key,
//                                 new cpu_thread_context_t
// #else
//             cpu_thread_context_t::current() =
// #endif
//                                 {
//                                     .tid = tid,
//                                     .bid = bid,
//                                     .cid = cid,
//                                 }
// #ifdef __APPLE__
//             );
// #else
//             ;
// #endif

//             size_t cpu_id = id;
// #if WIN32
//             SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 <<
//             cpu_id);
// #elif defined(__APPLE__)
//             thread_affinity_policy_data_t policy = {(int)cpu_id};
//             thread_policy_set(pthread_mach_thread_np(pthread_self()),
//                               THREAD_AFFINITY_POLICY,
//                               (thread_policy_t)&policy,
//                               THREAD_AFFINITY_POLICY_COUNT);
// #else
//             cpu_set_t cpuset;
//             CPU_ZERO(&cpuset);
//             CPU_SET(cpu_id, &cpuset);
//             pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t),
//             &cpuset);
// #endif

//             dynamic_shape_t<ntt::distributed::topology_levels> mesh_index;
//             mesh_index[0] = cid;
//             mesh_index[1] = bid;
//             mesh_index[2] = tid;
//             auto local_data_lhs = reinterpret_cast<float
//             *>(global_local_data_ptr(mesh_index)[0]); auto local_data_rhs =
//             reinterpret_cast<float *>(nncase::ntt::runtime::thread_alloc(M *
//             N * sizeof(float), 8));

//             // shard
//             tensor_view<float, fixed_shape_v<M, N>> tv_in(std::span<float, M
//             * N>(p_in, M * N)); sharded_tensor_view<float, fixed_shape_v<M,
//             N>, sharding<mesh<topology::thread, cdims, bdims, tdims>, B, I,
//             I>> stv_src(std::span<float, M * N>(local_data_lhs, M * N));
//             reshard(tv_in, stv_src);
//             EXPECT_TRUE(NttTest::compare_tensor(tv_in, stv_src.local()));
//             distributed::topology_synchronize();

//             // reshard
//             constexpr size_t split_num = cdims * bdims * tdims;
//             sharded_tensor_view<float, fixed_shape_v<M, N>,
//             sharding<mesh<topology::thread, cdims, bdims, tdims>, B, S<0, 1,
//             2>, I>> stv_dst(std::span<float, M * N /
//             split_num>(local_data_rhs, M * N / split_num)); reshard(stv_src,
//             stv_dst);

//             // unshard
//             tensor_view<float, fixed_shape_v<M, N>> tv_out(std::span<float, M
//             * N>(p_out, M * N)); reshard(stv_dst, tv_out);
//             EXPECT_TRUE(NttTest::compare_tensor(tv_in, tv_out));

//             nncase::ntt::runtime::thread_free(local_data_rhs);
//         });
//     }

//     for (auto &t : threads)
//         t.join();

//     apply(global_local_data_ptr.shape(), [&](auto index) {
//         thread_free((void *)global_local_data_ptr(index)[0]);
//     });
// }

// TEST(CpuTest, reshard_2D_different_sharding_spec_split_broadcast) {
//     // init
//     constexpr size_t M = 512;
//     constexpr size_t N = 1024;
//     using tensor_type = ntt::tensor<float, ntt::fixed_shape_v<M, N>>;
//     std::unique_ptr<tensor_type> ntt_input(new tensor_type);
//     NttTest::init_tensor(*ntt_input, -2.f, 2.f);
//     auto p_in = reinterpret_cast<float *>(ntt_input->elements().data());

// #ifdef __APPLE__
//     pthread_key_t cpu_thread_context_key_ = {};
//     pthread_key_create(&cpu_thread_context_key_, [](void *ptr) {
//         delete (nncase::ntt::runtime::cpu_thread_context_t *)ptr;
//     });
//     cpu_thread_context_key = cpu_thread_context_key_;
// #endif

//     constexpr size_t cdims = ntt::distributed::cdim();
//     constexpr size_t bdims = ntt::distributed::bdim();
//     constexpr size_t tdims = ntt::distributed::tdim();
//     constexpr size_t num = cdims * bdims * tdims;

//     apply(global_local_data_ptr.shape(), [&](auto index) {
//         global_local_data_ptr(index)[0] =
//         (uintptr_t)(nncase::ntt::runtime::thread_alloc(M * N * sizeof(float),
//         8)); global_local_data_ptr(index)[1] =
//         global_local_data_ptr(index)[0] + M * N * sizeof(float);
//     });

//     std::vector<std::thread> threads;
//     for (size_t id = 0; id < num; id++) {
//         threads.emplace_back([cdims, bdims, tdims, id, M, N, p_in] {
//             size_t cid = id / (bdims * tdims);
//             size_t bid = id % (bdims * tdims) / tdims;
//             size_t tid = id % (bdims * tdims) % tdims;
// #ifdef __APPLE__
//             pthread_setspecific(
//                 cpu_thread_context_key,
//                 new cpu_thread_context_t
// #else
//             cpu_thread_context_t::current() =
// #endif
//                 {
//                     .tid = tid,
//                     .bid = bid,
//                     .cid = cid,
//                 }
// #ifdef __APPLE__
//             );
// #else
//             ;
// #endif

//             size_t cpu_id = id;
// #if WIN32
//             SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 <<
//             cpu_id);
// #elif defined(__APPLE__)
//             thread_affinity_policy_data_t policy = {(int)cpu_id};
//             thread_policy_set(pthread_mach_thread_np(pthread_self()),
//                               THREAD_AFFINITY_POLICY,
//                               (thread_policy_t)&policy,
//                               THREAD_AFFINITY_POLICY_COUNT);
// #else
//             cpu_set_t cpuset;
//             CPU_ZERO(&cpuset);
//             CPU_SET(cpu_id, &cpuset);
//             pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t),
//             &cpuset);
// #endif
//             dynamic_shape_t<ntt::distributed::topology_levels> mesh_index;
//             mesh_index[0] = cid;
//             mesh_index[1] = bid;
//             mesh_index[2] = tid;
//             auto local_data_lhs = reinterpret_cast<float
//             *>(global_local_data_ptr(mesh_index)[0]); auto local_data_rhs =
//             reinterpret_cast<float *>(nncase::ntt::runtime::thread_alloc(M *
//             N * sizeof(float), 8));

//             // shard(slice)
//             constexpr size_t split_num = cdims * bdims * tdims;
//             tensor_view<float, fixed_shape_v<M, N>> tv(std::span<float, M *
//             N>(p_in, M * N)); sharded_tensor_view<float, fixed_shape_v<M, N>,
//             sharding<mesh<topology::thread, cdims, bdims, tdims>, B, S<0, 1,
//             2>, I>> stv_src(std::span<float, M * N /
//             split_num>(local_data_lhs, M * N / split_num));
//             sharded_tensor_view<float, fixed_shape_v<M, N>,
//             sharding<mesh<topology::thread, cdims, bdims, tdims>, B, I, I>>
//             stv_dst(std::span<float, M * N>(local_data_rhs, M * N));

//             // shard
//             reshard(tv, stv_src);
//             distributed::topology_synchronize();

//             // reshard
//             reshard(stv_src, stv_dst);
//             EXPECT_TRUE(NttTest::compare_tensor(tv, stv_dst.local()));

//             nncase::ntt::runtime::thread_free(local_data_rhs);
//         });
//     }

//     for (auto &t : threads)
//         t.join();

//     apply(global_local_data_ptr.shape(), [&](auto index) {
//         thread_free((void *)global_local_data_ptr(index)[0]);
//     });
// }

// template <typename T>
// void dump_tensor(std::string &info, T &t) {
//     std::cout << info << ":";
//     apply(t.shape(), [&](auto index) {
//         std::cout << t(index) << " ";
//     });
//     std::cout << std::endl;
// }

TEST(CpuTest, reshard_2D_different_sharding_spec_different_split_axis) {
    // init
    constexpr size_t M = 512;
    constexpr size_t N = 1024;
    auto ntt_input = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    auto ntt_output = ntt::make_tensor<float>(ntt::fixed_shape_v<M, N>);
    NttTest::init_tensor(ntt_input, -2.f, 2.f);
    NttTest::init_tensor(ntt_output, -2.f, 2.f);
    auto p_in = reinterpret_cast<float *>(ntt_input.elements().data());
    auto p_out = reinterpret_cast<float *>(ntt_output.elements().data());

#ifdef __APPLE__
    pthread_key_t cpu_thread_context_key_ = {};
    pthread_key_create(&cpu_thread_context_key_, [](void *ptr) {
        delete (nncase::ntt::runtime::cpu_thread_context_t *)ptr;
    });
    cpu_thread_context_key = cpu_thread_context_key_;
#endif

    constexpr size_t cdims = ntt::distributed::cdim();
    constexpr size_t bdims = ntt::distributed::bdim();
    constexpr size_t tdims = ntt::distributed::tdim();
    constexpr size_t num = cdims * bdims * tdims;

    ntt::apply(
        ntt::distributed::detail::global_local_data_ptr.shape(),
        [&](auto index) {
            ntt::distributed::detail::global_local_data_ptr(index)(0_dim) =
                (uintptr_t)(ntt::runtime::thread_alloc(M * N * sizeof(float),
                                                       8));
            ntt::distributed::detail::global_local_data_ptr(index)(1_dim) =
                ntt::distributed::detail::global_local_data_ptr(index)(0_dim) +
                M * N * sizeof(float);
        });

    std::vector<std::thread> threads;
    for (size_t id = 0; id < num; id++) {
        threads.emplace_back([cdims, bdims, tdims, id, M, N, p_in, p_out] {
            size_t cid = id / (bdims * tdims);
            size_t bid = id % (bdims * tdims) / tdims;
            size_t tid = id % (bdims * tdims) % tdims;
#ifdef __APPLE__
            pthread_setspecific(cpu_thread_context_key,
                                new cpu_thread_context_t
#else
            cpu_thread_context_t::current() =
#endif
                                {
                                    .tid = tid,
                                    .bid = bid,
                                    .cid = cid,
                                }
#ifdef __APPLE__
            );
#else
            ;
#endif

            size_t cpu_id = id;
#if WIN32
            SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << cpu_id);
#elif defined(__APPLE__)
            thread_affinity_policy_data_t policy = {(int)cpu_id};
            thread_policy_set(pthread_mach_thread_np(pthread_self()),
                              THREAD_AFFINITY_POLICY, (thread_policy_t)&policy,
                              THREAD_AFFINITY_POLICY_COUNT);
#else
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(cpu_id, &cpuset);
            pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif

            // shard(slice)
            constexpr size_t split_num = bdims * tdims;
            auto tv_in = ntt::make_tensor_view_from_address(
                p_in, ntt::fixed_shape_v<M, N>);
            auto tv_out = ntt::make_tensor_view_from_address(
                p_out, ntt::fixed_shape_v<M, N>);

            const auto program_ids = make_shape(cid, bid, tid);
            float *local_data_lhs = reinterpret_cast<float *>(
                ntt::distributed::detail::global_local_data_ptr(program_ids)(
                    0_dim));
            auto local_data_rhs = reinterpret_cast<float *>(
                nncase::ntt::runtime::thread_alloc(M * N * sizeof(float), 8));

            using mesh_type =
                ntt::distributed::mesh<ntt::distributed::topology::thread,
                                       cdims, bdims, tdims>;
            auto sharding_src = ntt::distributed::make_sharding<mesh_type>(
                ntt::distributed::shard_policy::S<1, 2>(),
                ntt::distributed::shard_policy::B);
            auto sharding_dst = ntt::distributed::make_sharding<mesh_type>(
                ntt::distributed::shard_policy::B,
                ntt::distributed::shard_policy::S<1, 2>());

            auto shape = fixed_shape_v<M, N>;
            auto stv_src =
                ntt::distributed::make_sharded_tensor_view_from_address(
                    local_data_lhs, shape, sharding_src,
                    ntt::fixed_strides_v<N, 1>);
            auto stv_dst =
                ntt::distributed::make_sharded_tensor_view_from_address(
                    local_data_rhs, shape, sharding_dst,
                    ntt::fixed_strides_v<N / split_num, 1>);

            // shard
            reshard(tv_in, stv_src);
            distributed::topology_synchronize();

            // reshard
            reshard(stv_src, stv_dst);

            // unshard
            reshard(stv_dst, tv_out);

            nncase::ntt::runtime::thread_free(local_data_rhs);
        });
    }

    for (auto &t : threads)
        t.join();

    EXPECT_TRUE(NttTest::compare_tensor(ntt_input, ntt_output));

    ntt::apply(ntt::distributed::detail::global_local_data_ptr.shape(),
               [&](auto index) {
                   thread_free(
                       (void *)ntt::distributed::detail::global_local_data_ptr(
                           index)(0_dim));
               });
}

// TEST(CpuTest, reshard_reshape) {
//     // init
//     constexpr size_t M_LHS = 512;
//     constexpr size_t N_LHS = 1024;
//     constexpr size_t M_RHS = 512;
//     constexpr size_t N_RHS = 4;
//     constexpr size_t K_RHS = 256;

//     using tensor_type_lhs = ntt::tensor<float, ntt::fixed_shape_v<M_LHS,
//     N_LHS>>; std::unique_ptr<tensor_type_lhs> ntt_input(new tensor_type_lhs);

//     using tensor_type_rhs = ntt::tensor<float, ntt::fixed_shape_v<M_RHS,
//     N_RHS, K_RHS>>; std::unique_ptr<tensor_type_rhs> ntt_output(new
//     tensor_type_rhs); NttTest::init_tensor(*ntt_input, -2.f, 2.f);
//     NttTest::init_tensor(*ntt_output, -2.f, 2.f);
//     auto p_in = reinterpret_cast<float *>(ntt_input->elements().data());
//     auto p_out = reinterpret_cast<float *>(ntt_output->elements().data());

// #ifdef __APPLE__
//     pthread_key_t cpu_thread_context_key_ = {};
//     pthread_key_create(&cpu_thread_context_key_, [](void *ptr) { delete
//     (nncase::ntt::runtime::cpu_thread_context_t *)ptr; });
//     cpu_thread_context_key = cpu_thread_context_key_;
// #endif

//     constexpr size_t cdims = ntt::distributed::cdim();
//     constexpr size_t bdims = ntt::distributed::bdim();
//     constexpr size_t tdims = ntt::distributed::tdim();
//     constexpr size_t num = cdims * bdims * tdims;
//     std::vector<std::thread> threads;
//     for (size_t id = 0; id < num; id++) {
//         threads.emplace_back([cdims, bdims, tdims, id, M_LHS, N_LHS, M_RHS,
//         N_RHS, K_RHS, p_in, p_out] {
//             size_t cid = id / (bdims * tdims);
//             size_t bid = id % (bdims * tdims) / tdims;
//             size_t tid = id % (bdims * tdims) % tdims;
// #ifdef __APPLE__
//             pthread_setspecific(cpu_thread_context_key,
//                                 new cpu_thread_context_t
// #else
//             cpu_thread_context_t::current() =
// #endif
//                                 {
//                                     .tid = tid,
//                                     .bid = bid,
//                                     .cid = cid,
//                                 }
// #ifdef __APPLE__
//             );
// #else
//             ;
// #endif

//             size_t cpu_id = id;
// #if WIN32
//             SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 <<
//             cpu_id);
// #elif defined(__APPLE__)
//             thread_affinity_policy_data_t policy = {(int)cpu_id};
//             thread_policy_set(pthread_mach_thread_np(pthread_self()),
//                               THREAD_AFFINITY_POLICY,
//                               (thread_policy_t)&policy,
//                               THREAD_AFFINITY_POLICY_COUNT);
// #else
//             cpu_set_t cpuset;
//             CPU_ZERO(&cpuset);
//             CPU_SET(cpu_id, &cpuset);
//             pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t),
//             &cpuset);
// #endif
//             auto local_data = reinterpret_cast<float
//             *>(nncase::ntt::runtime::thread_alloc(2 * M_LHS * N_LHS *
//             sizeof(float), 8));

//             // shard(slice)
//             constexpr size_t split_num = bdims * tdims;
//             tensor_view<float, fixed_shape_v<M_LHS, N_LHS>>
//             tv_in(std::span<float, M_LHS * N_LHS>(p_in, M_LHS * N_LHS));
//             sharded_tensor_view<float, fixed_shape_v<M_LHS, N_LHS>,
//             sharding<mesh<topology::thread, cdims, bdims, tdims>, B, S<1, 2>,
//             I>> stv_src(std::span<float, M_LHS * N_LHS /
//             split_num>(local_data, M_LHS * N_LHS / split_num));
//             sharded_tensor_view<float, fixed_shape_v<M_RHS, N_RHS, K_RHS>,
//             sharding<mesh<topology::thread, cdims, bdims, tdims>, B, I, I,
//             S<1, 2>>> stv_dst(std::span<float, M_RHS * N_RHS * K_RHS /
//             split_num>(local_data + M_LHS * N_LHS, M_LHS * N_LHS /
//             split_num)); tensor_view<float, fixed_shape_v<M_RHS, N_RHS,
//             K_RHS>> tv_out(std::span<float, M_RHS * N_RHS * K_RHS>(p_out,
//             M_RHS * N_RHS * K_RHS));

//             reshard(tv_in, stv_src);
//             reshard(stv_src, stv_dst);
//             reshard(stv_dst, tv_out);
//             auto reshaped_out = tv_out.reshape(fixed_shape_v<M_LHS,
//             N_LHS>{}); EXPECT_TRUE(NttTest::compare_tensor(tv_in,
//             reshaped_out));

//             nncase::ntt::runtime::thread_free(local_data);
//         });
//     }

//     for (auto &t : threads)
//         t.join();
// }

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}