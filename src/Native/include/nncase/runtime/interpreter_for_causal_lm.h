// /* Copyright 2019-2021 Canaan Inc.
//  *
//  * Licensed under the Apache License, Version 2.0 (the "License");
//  * you may not use this file except in compliance with the License.
//  * You may obtain a copy of the License at
//  *
//  *     http://www.apache.org/licenses/LICENSE-2.0
//  *
//  * Unless required by applicable law or agreed to in writing, software
//  * distributed under the License is distributed on an "AS IS" BASIS,
//  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  * See the License for the specific language governing permissions and
//  * limitations under the License.
//  */
// #pragma once
// #include "allocator.h"
// #include "dump_manager.h"
// #include "model.h"
// #include "runtime_module.h"
// #include "runtime_tensor.h"
// #include <memory>
// #include <nncase/runtime/attention_kv_cache.h>
// #include <nncase/runtime/interpreter.h>
// #include <nncase/runtime/stream.h>
// #include <nncase/shape.h>
// #include <nncase/tensor.h>
// #include <nncase/type.h>
// #include <unordered_map>
// #include <variant>

// BEGIN_NS_NNCASE_RUNTIME

// class NNCASE_API interpreter_for_causal_lm : public interpreter {
//   public:
//     interpreter_for_causal_lm() noexcept;
//     interpreter_for_causal_lm(interpreter_for_causal_lm &) = delete;
//     interpreter_for_causal_lm(interpreter_for_causal_lm &&) = default;
// };

// END_NS_NNCASE_RUNTIME