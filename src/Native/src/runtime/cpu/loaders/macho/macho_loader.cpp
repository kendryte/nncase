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
#include "macho_loader.h"
#include <cstdint>
#include <mach-o/dyld.h>
#include <nncase/runtime/result.h>
#include <nncase/runtime/span_reader.h>
#include <sys/mman.h>

using namespace nncase::runtime;

macho_loader::~macho_loader() {
    if (!NSUnLinkModule(reinterpret_cast<NSModule>(mod_),
                        NSUNLINKMODULE_OPTION_NONE)) {
        // throw std::runtime_error("NSUnLinkModule failed");
    }

    if (!NSDestroyObjectFileImage(reinterpret_cast<NSObjectFileImage>(ofi_))) {
        // throw std::runtime_error("NSDestroyObjectFileImage failed");
    }
}

void macho_loader::load(gsl::span<const gsl::byte> macho) {
    if (NSCreateObjectFileImageFromMemory(
            macho.data(), macho.size_bytes(),
            reinterpret_cast<NSObjectFileImage *>(&ofi_)) !=
        NSObjectFileImageSuccess) {
        throw std::runtime_error("NSCreateObjectFileImageFromMemory failed");
    }
    mod_ = reinterpret_cast<NSModule>(
        NSLinkModule(reinterpret_cast<NSObjectFileImage>(ofi_), "he_he",
                     NSLINKMODULE_OPTION_NONE));
    if (mod_ == NULL) {
        throw std::runtime_error("NSLinkModule failed");
    }

    sym_ = reinterpret_cast<NSSymbol>(NSLookupSymbolInModule(
        reinterpret_cast<NSModule>(mod_), "_kernel_entry"));
    if (sym_ == NULL) {
        throw std::runtime_error("NSLookupSymbolInModule failed");
    }
}

void *macho_loader::entry() const noexcept {
    return NSAddressOfSymbol(reinterpret_cast<NSSymbol>(sym_));
}
