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
#include "runtime_function.h"
#include <cuda_runtime_api.h>
#include <nncase/ntt/arch/cuda/runtime.h>
#include <nncase/runtime/dbg.h>
#include <nncase/runtime/interpreter.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/type_serializer.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::cuda;
using namespace nncase::ntt::runtime;

#define WARP_SIZE 32

result<void> cuda_runtime_function::run(std::byte *output_data) noexcept {
    auto enable_profiling = module()
                                .interp()
                                .options()
                                .get_scalar_opt<uint8_t>("enable_profiling")
                                .or_(false);
    for (size_t cid = 0; cid < module().cdim(); cid++) {
        CHECK_CUDA(cudaSetDevice(cid));

        cuda_block_entry_params_t *params;
        CHECK_CUDA(cudaMallocHost((void **)&params,
                                  sizeof(cuda_block_entry_params_t)));

        cuda_block_entry_params_t src_params{
            .tdim = module().tdim(),
            .bdim = module().bdim(),
            .cdim = module().cdim(),
            .bid = 0,
            .cid = cid,
            .cpu_id_offset = cid * module().bdim() * module().tdim(),
            .enable_profiling = enable_profiling,
            .input_descs = this->input_descs_.data(),
            .output_descs = this->output_descs_.data(),
            .rdata = module().rdata(),
            .output = output_data,
            .thread_local_rdata_header = module().thread_local_rdata_header(
                cid * module().bdim() * module().tdim()),
            .thread_local_rdata = module().thread_local_rdata_content(),
            .block_local_rdata = module().block_local_rdata(),
            .thread_local_data = thread_local_data(cid * module().bdim()),
            .block_local_data = block_local_data(cid * module().bdim()),
            .profile_records =
                enable_profiling
                    ? thread_local_profile_records(cid * module().bdim())
                    : std::span<ntt::runtime::profile_record>{},
            .profile_record_counts =
                enable_profiling
                    ? thread_local_profile_record_counts(cid * module().bdim())
                          .data()
                    : nullptr,
        };
        memcpy(params, &src_params, sizeof(cuda_block_entry_params_t));

        void *args[] = {&params};
        CHECK_CUDA(cudaLaunchKernel(
            (const void *)block_entry_, dim3(module().bdim()),
            dim3(module().tdim() * WARP_SIZE), args, 0, nullptr));
    }

    for (size_t cid = 0; cid < module().cdim(); cid++) {
        CHECK_CUDA(cudaSetDevice(cid));
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    return ok();
}
