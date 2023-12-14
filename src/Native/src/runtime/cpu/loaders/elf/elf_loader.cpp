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
#include <sys/mman.h>

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

void elf_loader::load(gsl::span<const gsl::byte> elf) {
    ctx_.elf = (void *)elf.data();
    el_init(&ctx_);

    buffer_ = (gsl::byte *)malloc(ctx_.memsz + ctx_.align);
    image_ =
        (gsl::byte *)(((size_t)buffer_ + (ctx_.align - 1)) & ~(ctx_.align - 1));

#if defined(__linux__)
    mprotect(image_, ctx_.memsz, PROT_READ | PROT_WRITE | PROT_EXEC);
#endif
    ctx_.base_load_vaddr = ctx_.base_load_paddr = (uintptr_t)image_;
    el_load(&ctx_, alloccb);
    el_relocate(&ctx_);
}

void *elf_loader::entry() const noexcept { return image_ + ctx_.ehdr.e_entry; }
