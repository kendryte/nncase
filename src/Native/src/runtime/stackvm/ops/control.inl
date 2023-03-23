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

NNCASE_STACKVM_DISPATCH_BEGIN(NOP)
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(BR)
try_(pc_relative(op.target));
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(BR_TRUE)
auto value = stack_.pop();
if (value.as_i())
    try_(pc_relative(op.target));
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(BR_FALSE)
auto value = stack_.pop();
if (!value.as_i())
    try_(pc_relative(op.target));
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(RET)
try_var(ret_addr, frames_.pop());
if (frames_.empty()) {
    return ok();
} else {
    try_(pc(ret_addr));
}
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(CALL)
return err(std::errc::not_supported);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(ECALL)
return err(std::errc::not_supported);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(EXTCALL)
try_(visit(op));
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(CUSCALL)
try_(visit(op));
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(THROW)
return err(std::errc::not_supported);
NNCASE_STACKVM_DISPATCH_END()

NNCASE_STACKVM_DISPATCH_BEGIN(BREAK)
return err(std::errc::not_supported);
NNCASE_STACKVM_DISPATCH_END()
