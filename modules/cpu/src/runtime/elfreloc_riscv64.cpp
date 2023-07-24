#include "elfload.h"

#if defined(__riscv)

#define R_riscv64_NONE     0
#define R_riscv64_RELATIVE 3
#define R_riscv64_JUMP_SLOT 5

el_status el_applyrela(el_ctx *ctx, Elf_RelA *rel)
{
    uint64_t *p = (uint64_t*) (rel->r_offset + ctx->base_load_vaddr);
    uint32_t type = ELF_R_TYPE(rel->r_info);
    EL_DEBUG("rv\n");

    switch (type) {
        case R_riscv64_NONE: break;
        case R_riscv64_RELATIVE:
            EL_DEBUG("Applying R_riscv64_RELATIVE reloc @%p\n", p);
            *p = rel->r_addend + ctx->base_load_vaddr;
            break;
        case R_riscv64_JUMP_SLOT:
            EL_DEBUG("Applying R_riscv64_JUMP_SLOT reloc @%p\n", p);
            break;
        default:
            EL_DEBUG("Bad relocation %u\n", type);
            return EL_BADREL;

    }

    return EL_OK;
}

#endif
