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
#include <nncase/ntt/arch/cpu/profiling.h>
#include <nncase/ntt/arch/cpu/runtime.h>
#include <nncase/runtime/dbg.h>
#include <nncase/runtime/interpreter.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/type_serializer.h>
#include <thread>
#include <vector>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::cpu;
using namespace nncase::ntt::runtime;

result<void> cpu_runtime_function::run(std::byte *output_data) noexcept {
    std::vector<std::thread> blocks;
    timer_record timer_records[24];
    try_var(enable_profiling,
            module().interp().options().get_scalar_opt<uint8_t>(
                "enable_profiling"));
    for (size_t cid = 0; cid < module().cdim(); cid++) {
        for (size_t bid = 0; bid < module().bdim(); bid++) {
            auto linear_bid = cid * module().bdim() + bid;
            auto tid_offset = linear_bid * module().tdim();
            blocks.emplace_back([cid, bid, linear_bid, tid_offset,
                                 enable_profiling, timer_records, output_data,
                                 this] {
                cpu_block_entry_params_t block_entry_params{
                    .tdim = module().tdim(),
                    .bdim = module().bdim(),
                    .cdim = module().cdim(),
                    .bid = bid,
                    .cid = cid,
                    .cpu_id_offset = tid_offset,
                    .input_descs = this->input_descs_.data(),
                    .output_descs = this->output_descs_.data(),
                    .rdata = module().rdata(),
                    .output = output_data,
                    .enable_profiling = enable_profiling,
                    .timer_records = const_cast<timer_record *>(
                        &timer_records[cid * module().bdim() * module().tdim() +
                                       bid * module().tdim()]),
                    .local_rdata_header =
                        module().local_rdata_header(tid_offset),
                    .local_rdata = module().local_rdata_content(),
                    .block_local_data = block_local_data(linear_bid),
#ifdef __APPLE__
                    .cpu_thread_context_key = module().cpu_thread_context_key(),
#endif
                };

                block_entry_(block_entry_params);
            });
        }
    }

    for (auto &block : blocks) {
        block.join();
    }

    return ok();
}
