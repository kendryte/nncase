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
#include <stdexcept>
#include <thread>
#include <vector>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::cpu;
using namespace nncase::ntt::runtime;

result<void> cpu_runtime_function::run(std::span<std::byte *> params) noexcept {
    std::vector<std::thread> blocks;
    timer_record timer_records[24];
    try_var(en_profiler,
            module().interp().options().get_scalar_opt<uint8_t>("en_profiler"));
    for (size_t cid = 0; cid < module().cdim(); cid++) {
        for (size_t bid = 0; bid < module().bdim(); bid++) {
            auto tid_offset = (cid * module().bdim() + bid) * module().tdim();
            blocks.emplace_back([cid, bid, params, tid_offset, en_profiler,
                                 timer_records, this] {
                cpu_block_entry_params_t block_entry_params{
                    .tdim = module().tdim(),
                    .bdim = module().bdim(),
                    .cdim = module().cdim(),
                    .bid = bid,
                    .cid = cid,
                    .cpu_id_offset = tid_offset,
                    .inouts = params.data(),
                    .rdata = module().rdata().data(),
                    .en_profiler = en_profiler,
                    .timer_records = const_cast<timer_record *>(timer_records),
                    .local_rdata_header =
                        module().local_rdata_header(tid_offset),
                    .local_rdata = module().local_rdata_content().data(),
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
