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
#include <nncase/runtime/dbg.h>
#include <nncase/value.h>

using namespace nncase;

result<void> tuple_node::copy_to(value_t dest) const noexcept {
    try_var(dest_tuple, dest.as<tuple>());
    CHECK_WITH_ERR(fields().size() == dest_tuple->fields().size(),
                   std::errc::invalid_argument);
    for (size_t i = 0; i < fields().size(); i++) {
        try_(fields()[i]->copy_to(dest_tuple->fields()[i]));
    }
    return ok();
}
