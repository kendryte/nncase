#ifndef ELFLOAD_H
#define ELFLOAD_H
#include "elf.h"
#include "elfarch.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef DEBUG
#include <stdio.h>
#define EL_DEBUG(...) printf(__VA_ARGS__)
#else
#define EL_DEBUG(...)                                                          \
    do {                                                                       \
    } while (0)
#endif

typedef enum {
    EL_OK = 0,

    EL_EIO,
    EL_ENOMEM,

    EL_NOTELF,
    EL_WRONGBITS,
    EL_WRONGENDIAN,
    EL_WRONGARCH,
    EL_WRONGOS,
    EL_NOTEXEC,
    EL_NODYN,
    EL_BADREL,

} el_status;

typedef struct el_ctx {
    bool (*pread)(struct el_ctx *ctx, void *dest, size_t nb, size_t offset);

    /* base_load_* -> address we are actually going to load at
     */
    Elf_Addr base_load_paddr, base_load_vaddr;

    /* size in memory of binary */
    Elf_Addr memsz;

    /* required alignment */
    Elf_Addr align;

    /* ELF header */
    Elf_Ehdr ehdr;

    /* Offset of dynamic table (0 if not ET_DYN) */
    Elf_Off dynoff;
    /* Size of dynamic table (0 if not ET_DYN) */
    Elf_Addr dynsize;

    void *elf;
} el_ctx;

el_status el_pread(el_ctx *ctx, void *def, size_t nb, size_t offset);

el_status el_init(el_ctx *ctx);
typedef void *(*el_alloc_cb)(el_ctx *ctx, Elf_Addr phys, Elf_Addr virt,
                             Elf_Addr size);

el_status el_load(el_ctx *ctx, el_alloc_cb alloccb);

/* find the next phdr of type \p type, starting at \p *i.
 * On success, returns EL_OK with *i set to the phdr number, and the phdr loaded
 * in *phdr.
 *
 * If the end of the phdrs table was reached, *i is set to -1 and the contents
 * of *phdr are undefined
 */
el_status el_findphdr(el_ctx *ctx, Elf_Phdr *phdr, uint32_t type, unsigned *i);

/* Relocate the loaded executable */
el_status el_relocate(el_ctx *ctx);

/* find a dynamic table entry
 * returns the entry on success, dyn->d_tag = DT_NULL on failure
 */
el_status el_finddyn(el_ctx *ctx, Elf_Dyn *dyn, uint32_t type);

typedef struct {
    Elf_Off tableoff;
    Elf_Addr tablesize;
    Elf_Addr entrysize;
} el_relocinfo;

/* find all information regarding relocations of a specific type.
 *
 * pass DT_REL or DT_RELA for type
 * sets ri->entrysize = 0 if not found
 */
el_status el_findrelocs(el_ctx *ctx, el_relocinfo *ri, uint32_t type);

#endif