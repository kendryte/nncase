#pragma once
#include "elfload.h"
#include "hardware_def.h"
#include "method_table_impl.h"
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

typedef void (*entrypoint_t)(hardware_context_mt *hw_ctx_impl,
                             runtime_util_mt *rt_util_mt,
                             nncase_mt_t *nncase_mt_impl, uint8_t **inputs,
                             uint8_t *rdata);

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

    int invoke_elf(hardware_context_mt *hw_ctx_impl,
                   runtime_util_mt *rt_util_mt, nncase_mt_t *nncase_mt_impl,
                   uint8_t **inputs, uint8_t *rdata);

  private:
    void *ptr_;
    void *buf_;
    el_ctx ctx_;
};

END_NS_NNCASE_RT_MODULE