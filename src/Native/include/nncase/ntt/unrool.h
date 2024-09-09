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
#include <cstddef>
#include <utility>

namespace nncase::ntt {
#define THREAD(N, i, rets, input, Stride, op, isBinary)                        \
    if constexpr (UnRoolNum > N) {                                             \
        if constexpr (isBinary)                                                \
            rets[N] = op(rets[N], input[(i + N) * Stride]);                    \
        else                                                                   \
            rets[(i + N) * Stride] = op(input[(i + N) * Stride]);              \
    }

// 目前只支持到最大8维度展开，如果有需要可以扩展
#define UNROLL_LOOP(i, rets, input, Stride, op, isBinary)                      \
    THREAD(0, i, rets, input, Stride, op, isBinary)                            \
    THREAD(1, i, rets, input, Stride, op, isBinary)                            \
    THREAD(2, i, rets, input, Stride, op, isBinary)                            \
    THREAD(3, i, rets, input, Stride, op, isBinary)                            \
    THREAD(4, i, rets, input, Stride, op, isBinary)                            \
    THREAD(5, i, rets, input, Stride, op, isBinary)                            \
    THREAD(6, i, rets, input, Stride, op, isBinary)                            \
    THREAD(7, i, rets, input, Stride, op, isBinary)

template <template <typename T1, typename T2> class Op, class TElem,
          size_t UnRoolNum, size_t LoopCnt, size_t Stride>
auto loop_unrool(const TElem *input) {

    // static_assert(UnRoolNum <= LoopCnt, "UnRoolNum must be less than
    // LoopCnt");
    static_assert(UnRoolNum > 0, "UnRoolNum must be greater than zero");
    static_assert(UnRoolNum < 9, "UnRoolNum must be less than 9");
    TElem ret;
    TElem rets[UnRoolNum];

    Op<TElem, TElem> op;

    for (size_t i = 0; i < UnRoolNum; i++) {
        rets[i] = input[i * Stride];
    }

    constexpr size_t remainder = LoopCnt % UnRoolNum;
    constexpr size_t integer = LoopCnt - remainder;
    for (size_t i = UnRoolNum; i < integer; i += UnRoolNum) {
        UNROLL_LOOP(i, rets, input, Stride, op, true)
    }

    ret = rets[0];
    for (size_t i = 1; i < UnRoolNum; i++) {
        ret = op(ret, rets[i]);
    }

    for (size_t i = 0; i < remainder; i++) {
        ret = op(ret, input[(integer + i) * Stride]);
    }

    return ret;
}

template <template <typename T1, typename T2> class Op, class TElem,
          size_t UnRoolNum, size_t LoopCnt, size_t Stride>
auto loop_unrool(const TElem *input_a, const TElem *input_b) {

    Op<TElem, TElem> op;
    auto result_a = loop_unrool<Op, TElem, UnRoolNum, LoopCnt, Stride>(input_a);
    auto result_b = loop_unrool<Op, TElem, UnRoolNum, LoopCnt, Stride>(input_b);
    return op(result_a, result_b);
}

template <template <typename T> class Op, class TElem, size_t UnRoolNum,
          size_t LoopCnt, size_t Stride>
void loop_unrool(const TElem *input, TElem *output) {

    static_assert(UnRoolNum <= LoopCnt, "UnRoolNum must be less than LoopCnt");
    static_assert(UnRoolNum > 0, "UnRoolNum must be greater than zero");
    static_assert(UnRoolNum < 9, "UnRoolNum must be less than 9");
    TElem *ret = output;
    Op<TElem> op;

    constexpr size_t remainder = LoopCnt % UnRoolNum;
    constexpr size_t integer = LoopCnt - remainder;
    for (size_t i = 0; i < integer; i += UnRoolNum) {
        UNROLL_LOOP(i, ret, input, Stride, op, false)
    }

    for (size_t i = 0; i < remainder; i++) {
        ret[(integer + i) * Stride] = op(input[(integer + i) * Stride]);
    }
}
} // namespace nncase::ntt