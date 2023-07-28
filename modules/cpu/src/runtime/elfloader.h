#pragma once
#include "cpu_common.h"
#include "elfload.h"
#include <inttypes.h>
#include <math.h>
#include <nncase/runtime/cpu/compiler_defs.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(__linux__)
#include <sys/mman.h>
#endif

BEGIN_NS_NNCASE_RT_MODULE(cpu)

typedef void (*entrypoint_t)(size_t id, uint8_t **buffers,
                             nncase_mt_t *nncase_mt, void *data, const uint8_t *rdata);

class elfloader {
  public:
    elfloader(char *elf) {
        ctx_.pread = bpread;
        ctx_.elf = elf;
    }

    // typedef void (*entrypoint_t)(float (*op_t)(float), float *, float *,
    // int);

    static bool bpread(el_ctx *ctx, void *dest, size_t nb, size_t offset) {
        (void)ctx;

        memcpy(dest, (char *)ctx->elf + offset, nb);

        return true;
    }

    static void *alloccb(el_ctx *ctx, Elf_Addr phys, Elf_Addr virt,
                         Elf_Addr size) {
        (void)ctx;
        (void)phys;
        (void)size;
        return (void *)virt;
    }

    static void check(el_status stat, const char *expln) {
        if (stat) {
            fprintf(stderr, "%s: error %d\n", expln, stat);
            exit(1);
        }
    }

    int invoke_elf(size_t id, uint8_t **buffers, nncase_mt_t *nncase_mt,
                   void *data, const uint8_t *rdata);

  private:
    void *ptr_;
    void *buf_;
    el_ctx ctx_;
};

END_NS_NNCASE_RT_MODULE