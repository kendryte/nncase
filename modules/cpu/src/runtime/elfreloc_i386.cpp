#include "elfload.h"

#if defined(__i386__)

#define R_386_NONE     0
#define R_386_RELATIVE 8

el_status el_applyrel(el_ctx *ctx, Elf_Rel *rel)
{
    uint32_t *p = (uint32_t*) (rel->r_offset + ctx->base_load_vaddr);
    uint32_t type = ELF_R_TYPE(rel->r_info);
    uint32_t sym  = ELF_R_SYM(rel->r_info);

    switch (type) {
        case R_386_NONE: break;
        case R_386_RELATIVE:
            EL_DEBUG("Applying R_386_RELATIVE reloc @%p\n", p);
            *p += ctx->base_load_vaddr;
            break;
        default:
            EL_DEBUG("Bad relocation %u\n", type);
            return EL_BADREL;
    }

    return EL_OK;
}

#endif
