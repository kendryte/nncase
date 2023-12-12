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

BEGIN_NS_NNCASE_RUNTIME

class macho_loader {
  public:
    macho_loader() noexcept : image_(nullptr) {}
    ~macho_loader();

    void load(const gsl::byte *macho);
    void *entry() const noexcept;

  private:
    gsl::byte *image_;
    uint64_t image_size_;
    uint64_t pc_;
};

END_NS_NNCASE_RUNTIME
