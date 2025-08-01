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
#include <dlfcn.h>
#include <fcntl.h>
#include <mach-o/dyld.h>
#include <nncase/runtime/result.h>
#include <nncase/runtime/span_reader.h>
#include <sys/mman.h>

using namespace nncase::runtime;

#define THROW_SYS_IF_NOT(x)                                                    \
    if (!(x)) {                                                                \
        throw std::system_error(errno, std::system_category());                \
    }

macho_loader::~macho_loader() {
#if 0
    if (!NSUnLinkModule(reinterpret_cast<NSModule>(mod_),
                        NSUNLINKMODULE_OPTION_NONE)) {
        abort();
    }

    if (!NSDestroyObjectFileImage(reinterpret_cast<NSObjectFileImage>(ofi_))) {

        abort();
    }
#else
    if (mod_) {
        dlclose(mod_);
    }
#endif
}

void macho_loader::load(std::span<const std::byte> macho) {
#if 0
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
        reinterpret_cast<NSModule>(mod_), "_module_entry"));
    if (sym_ == NULL) {
        throw std::runtime_error("NSLookupSymbolInModule failed");
    }
#else
    char temp_path[] = "/tmp/nncase.function.cpu.XXXXXX";
    {
        auto func_file = mkstemp(temp_path);
        THROW_SYS_IF_NOT(func_file != -1);
        THROW_SYS_IF_NOT(write(func_file, (char *)macho.data(), macho.size()) !=
                         -1);
        THROW_SYS_IF_NOT(close(func_file) != -1);
    }

    mod_ = dlopen(temp_path, RTLD_NOW);
    if (!mod_) {
        throw std::runtime_error("dlopen error:" + std::string(dlerror()));
    }

    sym_ = dlsym(mod_, "block_entry");
    if (!sym_) {
        throw std::runtime_error("dlsym error:" + std::string(dlerror()));
    }
#endif
}

void macho_loader::load_from_file(std::string_view path) {
    mod_ = dlopen(path.data(), RTLD_NOW);
    if (!mod_) {
        throw std::runtime_error("dlopen error:" + std::string(dlerror()));
    }

    sym_ = dlsym(mod_, "block_entry");
    if (!sym_) {
        throw std::runtime_error("dlsym error:" + std::string(dlerror()));
    }
}

void *macho_loader::entry() const noexcept {
#if 0
    return NSAddressOfSymbol(reinterpret_cast<NSSymbol>(sym_));
#else
    return sym_;
#endif
}
