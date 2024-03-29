#ifndef ELFARCH_H
#define ELFARCH_H

#if defined(__i386__)
#define EM_THIS EM_386
#define EL_ARCH_USES_REL
#elif defined(__amd64__)
#define EM_THIS EM_AMD64
#define EL_ARCH_USES_RELA
#elif defined(__arm__)
#define EM_THIS EM_ARM
#elif defined(__aarch64__)
#define EM_THIS EM_AARCH64
#define EL_ARCH_USES_RELA
#define EL_ARCH_USES_REL
#elif defined(__riscv)
#define EM_THIS EM_RISCV
#define EL_ARCH_USES_RELA
#else
#error specify your ELF architecture
#endif

#if defined(__LP64__) || defined(__LLP64__)
#define ELFSIZE 64
#else
#define ELFSIZE 32
#endif

#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#define ELFDATATHIS ELFDATA2LSB
#else
#define ELFDATATHIS ELFDATA2MSB
#endif

#endif
