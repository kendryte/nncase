#include "elfloader.h"
#include "thread_pool.h"

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::cpu;

int elfloader::invoke_elf(hardware_context_mt *hw_ctx_impl,
                          runtime_util_mt *rt_util_mt,
                          nncase_mt_t *nncase_mt_impl, uint8_t **inputs,
                          uint8_t *rdata) {

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

#if defined(__riscv)
    Elf_Shdr sym_shdr;
    for (size_t i = 0; i < ctx_.ehdr.e_shnum; i++)
    {
        memcpy(&sym_shdr, (void*)((uintptr_t)ctx_.elf + ctx_.ehdr.e_shoff + ctx_.ehdr.e_shentsize*i), ctx_.ehdr.e_shentsize);
        if (sym_shdr.sh_type == SHT_SYMTAB) {
            break;
        }
    }
    
    Elf_Shdr str_shdr;
    memcpy(&str_shdr, (void*)((uintptr_t)ctx_.elf + ctx_.ehdr.e_shoff + ctx_.ehdr.e_shentsize*sym_shdr.sh_link), ctx_.ehdr.e_shentsize);
    
    size_t sym_num = sym_shdr.sh_size / sym_shdr.sh_entsize;
    Elf_Sym symbol;
    for (size_t i = 0; i < sym_num; i++)
    {
        memcpy(&symbol, (void*)((uintptr_t)ctx_.elf + sym_shdr.sh_offset + sym_shdr.sh_entsize*i), sym_shdr.sh_entsize);
        if (!strncmp((char*)((uintptr_t)ctx_.elf + str_shdr.sh_offset + symbol.st_name), "__global_pointer$", 19)) {
            break;
        }
    }
    asm volatile (
        "addi sp, sp, -8;\
        sd gp, 0(sp);\
        mv gp, %0\
        "
        :
        :"r"(symbol.st_value + (uintptr_t)buf_)
        :
    );
#endif
    ep(hw_ctx_impl, rt_util_mt, nncase_mt_impl, inputs, rdata);
#if defined(__riscv)

    asm volatile (
        "ld gp, 0(sp);\
        addi sp, sp, 8;\
        "
        :
        :
        :
    );
#endif
    free(ptr_);

    return 0;
}
