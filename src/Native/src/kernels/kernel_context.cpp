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
#include <nncase/kernels/kernel_context.h>
#ifdef NNCASE_OPENMP
#include <omp.h>
#endif

using namespace nncase;
using namespace nncase::kernels;

namespace {
struct default_kernel_context_holder {
    kernel_context ctx;

    default_kernel_context_holder() {
#ifdef NNCASE_OPENMP
        ctx.num_threads = (uint32_t)omp_get_max_threads();
#else
        ctx.num_threads = 1;
#endif
        ctx.dump_manager =
            std::shared_ptr<nncase::runtime::dump_manager>(nullptr);
    }
};
} // namespace

kernel_context &kernels::default_kernel_context() {
    static default_kernel_context_holder holder;
    return holder.ctx;
}
