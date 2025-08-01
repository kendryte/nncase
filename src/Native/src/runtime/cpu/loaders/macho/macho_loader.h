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
#include <nncase/compiler_defs.h>
#include <span>
#include <string_view>

BEGIN_NS_NNCASE_RUNTIME

class macho_loader {
  public:
    macho_loader() noexcept
        :
#if 0 
     ofi_(nullptr),
#endif
          mod_(nullptr),
          sym_(nullptr) {
    }
    ~macho_loader();

    void load(std::span<const std::byte> macho);
    void load_from_file(std::string_view path);
    void *entry() const noexcept;

  private:
#if 0
    void *ofi_;
#endif
    void *mod_;
    void *sym_;
};

END_NS_NNCASE_RUNTIME
