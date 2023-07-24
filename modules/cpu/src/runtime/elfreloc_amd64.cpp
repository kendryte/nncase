#include "elfload.h"

#if defined(__amd64__)

#define R_AMD64_NONE     0
#define R_AMD64_RELATIVE 8

el_status el_applyrela(el_ctx *ctx, Elf_RelA *rel)
{
    uint64_t *p = (uint64_t*) (rel->r_offset + ctx->base_load_vaddr);
    uint32_t type = ELF_R_TYPE(rel->r_info);

    switch (type) {
        case R_AMD64_NONE: break;
        case R_AMD64_RELATIVE:
            EL_DEBUG("Applying R_AMD64_RELATIVE reloc @%p\n", p);
            *p = rel->r_addend + ctx->base_load_vaddr;
            break;
        default:
            EL_DEBUG("Bad relocation %u\n", type);
            return EL_BADREL;

    }

    return EL_OK;
}

#endif
