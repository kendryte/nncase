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

#define CONV_IMPL(type)                                                        \
    auto value = stack_.pop();                                                 \
    if (!value.is_r())                                                         \
        stack_.push((type)value.as_i());                                       \
    else                                                                       \
        stack_.push((type)value.as_r())

NNCASE_STACKVM_DISPATCH_BEGIN(CONV_I1)
CONV_IMPL(int8_t);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(CONV_I2)
CONV_IMPL(int16_t);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(CONV_I4)
CONV_IMPL(int32_t);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(CONV_I)
CONV_IMPL(intptr_t);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(CONV_U1)
CONV_IMPL(uint8_t);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(CONV_U2)
CONV_IMPL(uint16_t);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(CONV_U4)
CONV_IMPL(uint32_t);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(CONV_U)
CONV_IMPL(uintptr_t);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(CONV_BR2)
CONV_IMPL(bfloat16);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(CONV_R4)
CONV_IMPL(float);
NNCASE_STACKVM_DISPATCH_END()
