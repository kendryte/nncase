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
#include <cstring>
#include <nncase/runtime/result.h>
#if defined(__linux__)
#include <chrono>
#include <dlfcn.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/mman.h>
#endif

using namespace nncase::runtime;

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

elf_loader::elf_loader() noexcept : buffer_(nullptr) { ctx_.pread = bpread; }

elf_loader::~elf_loader() {
    if (buffer_) {
        free(buffer_);
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
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);

        std::stringstream ss;
        ss << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S");
        std::string kernel_so = "/tmp/" + ss.str() + ".so";
        std::ofstream outputFile(kernel_so, std::ios::out | std::ios::binary);
        if (!outputFile) {
            std::cerr << "cannot create file:" << kernel_so << std::endl;
            throw std::runtime_error("cannot create file:" + kernel_so);
        }

        outputFile.write((char *)ctx_.elf, elf.size());

        if (!outputFile.good()) {
            std::cerr << "error writing file" << std::endl;
            outputFile.close();
            throw std::runtime_error("error writing file");
        }

        void *handle = dlopen(kernel_so.c_str(), RTLD_LAZY);
        if (!handle) {
            fprintf(stderr, "Error: %s\n", dlerror());
            exit(EXIT_FAILURE);
        }

        entry_ = dlsym(handle, "kernel_entry");
        const char *dlsym_error = dlerror();
        if (dlsym_error) {
            dlclose(handle);
            throw std::runtime_error("dlsym error:" + std::string(dlsym_error));
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
