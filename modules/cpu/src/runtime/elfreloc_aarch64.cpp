#include "elfload.h"

#if defined(__aarch64__)

#define R_AARCH64_NONE     0
#define R_AARCH64_RELATIVE 1027

el_status el_applyrela(el_ctx *ctx, Elf_RelA *rel)
{
    uintptr_t *p = (uintptr_t*) (rel->r_offset + ctx->base_load_paddr);
    uint32_t type = ELF_R_TYPE(rel->r_info);
    uint32_t sym  = ELF_R_SYM(rel->r_info);

    switch (type) {
        case R_AARCH64_NONE:
            EL_DEBUG("R_AARCH64_NONE\n");
            break;
        case R_AARCH64_RELATIVE:
            if (sym) {
                EL_DEBUG("R_AARCH64_RELATIVE with symbol ref!\n");
                return EL_BADREL;
            }

            EL_DEBUG("Applying R_AARCH64_RELATIVE reloc @%p\n", p);
            *p = rel->r_addend + ctx->base_load_vaddr;
            break;

        default:
            EL_DEBUG("Bad relocation %u\n", type);
            return EL_BADREL;

    }

    return EL_OK;
}

el_status el_applyrel(el_ctx *ctx, Elf_Rel *rel)
{
    uintptr_t *p = (uintptr_t*) (rel->r_offset + ctx->base_load_paddr);
    uint32_t type = ELF_R_TYPE(rel->r_info);
    uint32_t sym  = ELF_R_SYM(rel->r_info);

    switch (type) {
        case R_AARCH64_NONE:
            EL_DEBUG("R_AARCH64_NONE\n");
            break;
        case R_AARCH64_RELATIVE:
            if (sym) {
                EL_DEBUG("R_AARCH64_RELATIVE with symbol ref!\n");
                return EL_BADREL;
            }

            EL_DEBUG("Applying R_AARCH64_RELATIVE reloc @%p\n", p);
            *p += ctx->base_load_vaddr;
            break;

        default:
            EL_DEBUG("Bad relocation %u\n", type);
            return EL_BADREL;

    }

    return EL_OK;
}


#endif
