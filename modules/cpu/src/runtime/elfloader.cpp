#include "elfloader.h"
#include "thread_pool.h"

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::cpu;

int elfloader::invoke_elf(hardware_context_mt *hw_ctx_impl, runtime_util_mt *rt_util_mt,
            nncase_mt_t *nncase_mt_impl, uint8_t **inputs) {

    check(el_init(&ctx_), "initialising");

    // align to ctx.align
    ptr_ = malloc(ctx_.memsz + ctx_.align);
    buf_ = (void *)(((size_t)ptr_ + (ctx_.align - 1)) & ~(ctx_.align - 1));

#if defined(__linux__)
    if (mprotect(buf_, ctx_.memsz, PROT_READ | PROT_WRITE | PROT_EXEC)) {
        perror("mprotect");
        return 1;
    }
#endif

    ctx_.base_load_vaddr = ctx_.base_load_paddr = (uintptr_t)buf_;

    check(el_load(&ctx_, alloccb), "loading");
    check(el_relocate(&ctx_), "relocating");

    uintptr_t epaddr = ctx_.ehdr.e_entry + (uintptr_t)buf_;

    entrypoint_t ep = (entrypoint_t)epaddr;

    // printf("Binary entrypoint is %" PRIxPTR "; invoking %p\n",
    //        (uintptr_t)ctx_.ehdr.e_entry, (void *)epaddr);

    ep(hw_ctx_impl, rt_util_mt, nncase_mt_impl, inputs);

    free(ptr_);

    return 0;
}
