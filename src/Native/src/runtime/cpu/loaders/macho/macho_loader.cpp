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
#include <mach-o/loader.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <mach/thread_status.h>
#include <nncase/runtime/result.h>
#include <nncase/runtime/span_reader.h>
#include <sys/mman.h>

using namespace nncase::runtime;

macho_loader::~macho_loader() {
    if (image_) {
        mach_vm_deallocate(mach_task_self(), (mach_vm_address_t)image_,
                           image_size_);
    }
}

void macho_loader::load(const gsl::byte *macho) {
    auto header = reinterpret_cast<const mach_header_64 *>(macho);
    auto lc_base =
        reinterpret_cast<const load_command *>(macho + sizeof(mach_header_64));

    // 1. Calc image size
    uint64_t max_addr = 0;
    uint64_t max_length = 0;
    const load_command *cnt_lc = lc_base;
    for (size_t i = 0; i < header->ncmds; i++) {
        if (cnt_lc->cmd == LC_SEGMENT_64) {
            auto seg_lc = (const segment_command_64 *)cnt_lc;
            if (seg_lc->vmaddr >= max_addr + max_length) {
                max_addr = seg_lc->vmaddr;
                max_length = seg_lc->vmsize;
            }
        }

        cnt_lc = (const load_command *)((intptr_t)cnt_lc + cnt_lc->cmdsize);
    }

    // 2. Allocate image
    image_size_ = max_addr + max_length;
    mach_vm_allocate(mach_task_self(),
                     reinterpret_cast<mach_vm_address_t *>(&image_),
                     image_size_, VM_FLAGS_ANYWHERE);

    // 3. Handle load commands
    cnt_lc = lc_base;
    for (size_t i = 0; i < header->ncmds; i++) {
        if (cnt_lc->cmd == LC_SEGMENT_64) {
            auto seg_lc = (const segment_command_64 *)cnt_lc;
            memcpy(image_ + seg_lc->vmaddr, macho + seg_lc->fileoff,
                   seg_lc->filesize);
            mach_vm_protect(mach_task_self(),
                            (mach_vm_address_t)image_ + seg_lc->vmaddr,
                            seg_lc->vmsize, false, seg_lc->initprot);

            // 3.1 Handle sections
            auto sections = (const section_64 *)((intptr_t)seg_lc +
                                                 sizeof(segment_command_64));
            for (size_t j = 0; j < seg_lc->nsects; j++) {
                auto &section = sections[j];
                auto section_vaddr = image_ + section.addr;
                mach_vm_protect(mach_task_self(),
                                (mach_vm_address_t)section_vaddr, section.size,
                                false, PROT_READ | PROT_WRITE);
                memcpy(section_vaddr, macho + section.offset, section.size);
                mach_vm_protect(mach_task_self(),
                                (mach_vm_address_t)section_vaddr, section.size,
                                false, seg_lc->initprot);
            }
        } else if (cnt_lc->cmd == LC_UNIXTHREAD) {
            auto ut_lc = (const thread_command *)cnt_lc;
#if defined(__arm64__)
            auto state = (const arm_thread_state64_t *)((intptr_t)ut_lc +
                                                        sizeof(thread_command) +
                                                        sizeof(uint32_t) * 2);
            pc_ = arm_thread_state64_get_pc(*state);
#else
            auto state = (const x86_thread_state64_t *)((intptr_t)ut_lc +
                                                        sizeof(thread_command) +
                                                        sizeof(uint32_t) * 2);
            pc_ = state->__rip;
#endif
        }

        cnt_lc = (const load_command *)((intptr_t)cnt_lc + cnt_lc->cmdsize);
    }
}

void *macho_loader::entry() const noexcept { return image_ + pc_; }
