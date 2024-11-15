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
#include "elf_loader.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <nncase/runtime/result.h>
#if defined(__linux__)
#include <cstdio>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

using namespace nncase::runtime;

#define THROW_SYS_IF_NOT(x)                                                    \
    if (!(x)) {                                                                \
        throw std::system_error(errno, std::system_category());                \
    }

static bool bpread(el_ctx *ctx, void *dest, size_t nb, size_t offset) {
    (void)ctx;
    memcpy(dest, (char *)ctx->elf + offset, nb);
    return true;
}

static void *alloccb(el_ctx *ctx, Elf_Addr phys, Elf_Addr virt, Elf_Addr size) {
    (void)ctx;
    (void)phys;
    (void)size;
    return (void *)virt;
}

elf_loader::elf_loader() noexcept
    : buffer_(nullptr), image_(nullptr), handle_(nullptr) {
    ctx_.pread = bpread;
}

elf_loader::~elf_loader() {
    if (buffer_) {
        free(buffer_);
    }
    if (handle_) {
        dlclose(handle_);
    }
}

void elf_loader::load(std::span<const std::byte> elf) {
    ctx_.elf = (void *)elf.data();
    el_init(&ctx_);

    if (ctx_.ehdr.e_type == ET_EXEC) {
        buffer_ = (std::byte *)malloc(ctx_.memsz + ctx_.align);
        image_ = (std::byte *)(((size_t)buffer_ + (ctx_.align - 1)) &
                               ~(ctx_.align - 1));

#if defined(__linux__)
        mprotect(image_, ctx_.memsz, PROT_READ | PROT_WRITE | PROT_EXEC);
#endif
        ctx_.base_load_vaddr = ctx_.base_load_paddr = (uintptr_t)image_;
        el_load(&ctx_, alloccb);
        el_relocate(&ctx_);
#if defined(__linux__)
    } else if (ctx_.ehdr.e_type == ET_DYN) {
        char temp_path[] = "/tmp/nncase.function.cpu.XXXXXX";
        {
            auto func_file = mkstemp(temp_path);
            THROW_SYS_IF_NOT(func_file != -1);
            THROW_SYS_IF_NOT(write(func_file, (char *)ctx_.elf, elf.size()) !=
                             -1);
            THROW_SYS_IF_NOT(close(func_file) != -1);
        }

        handle_ = dlopen(temp_path, RTLD_NOW);
        if (!handle_) {
            throw std::runtime_error("dlopen error:" + std::string(dlerror()));
        }

        entry_ = dlsym(handle_, "block_entry");
        if (!entry_) {
            throw std::runtime_error("dlsym error:" + std::string(dlerror()));
        }
#endif
    } else {
        throw std::runtime_error("Unsupported ELF type");
    }
}

void *elf_loader::entry() const noexcept {
    if (ctx_.ehdr.e_type == ET_EXEC) {
        return image_ + ctx_.ehdr.e_entry;
    } else {
        return entry_;
    }
}
