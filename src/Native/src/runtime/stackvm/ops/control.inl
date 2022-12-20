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

#define NNCASE_STACKVM_DISPATCH_OP_NOP

#define NNCASE_STACKVM_DISPATCH_OP_BR try_(pc_relative(op.target));

#define NNCASE_STACKVM_DISPATCH_OP_BR_TRUE                                     \
    auto value = stack_.pop();                                                 \
    if (value.as_i())                                                          \
        try_(pc_relative(op.target));

#define NNCASE_STACKVM_DISPATCH_OP_BR_FALSE                                    \
    auto value = stack_.pop();                                                 \
    if (!value.as_i())                                                         \
        try_(pc_relative(op.target));

#define NNCASE_STACKVM_DISPATCH_OP_RET                                         \
    try_var(ret_addr, frames_.pop());                                          \
    if (frames_.empty()) {                                                     \
        return ok();                                                           \
    } else {                                                                   \
        try_(pc(ret_addr));                                                    \
    }

#define NNCASE_STACKVM_DISPATCH_OP_CALL return err(std::errc::not_supported);

#define NNCASE_STACKVM_DISPATCH_OP_ECALL return err(std::errc::not_supported);

#define NNCASE_STACKVM_DISPATCH_OP_EXTCALL try_(visit(op));

#define NNCASE_STACKVM_DISPATCH_OP_CUSCALL try_(visit(op));

#define NNCASE_STACKVM_DISPATCH_OP_THROW return err(std::errc::not_supported);

#define NNCASE_STACKVM_DISPATCH_OP_BREAK return err(std::errc::not_supported);
