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
#include "nncase/ntt/runtime.h"
#include "nncase/ntt/runtime/cpu_runtime.h"
#include "runtime_function.h"
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
    for (size_t cid = 0; cid < cdim_; cid++) {
        for (size_t bid = 0; bid < bdim_; bid++) {
            blocks.emplace_back([cid, bid, params, this] {
                cpu_block_entry_params_t block_entry_params{
                    .tdim = tdim_,
                    .bdim = bdim_,
                    .cdim = cdim_,
                    .bid = bid,
                    .cid = cid,
                    .cpu_id_offset = (cid * bdim_ + bid) * tdim_,
                    .inouts = params.data(),
                    .rdata = module().rdata().data(),
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
