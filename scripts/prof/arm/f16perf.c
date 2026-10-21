// fp16 vs fp32 SIMD throughput on the m9g core.
//   gcc -O3 -march=armv8.2-a+bf16+fp16 -o f16perf f16perf.c && ./f16perf
#include <stdio.h>
#include <time.h>

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}
static float buf[64];
static long IT = 2000000;

#define RUN(NAME, INSN, SUF)                                                 \
  do {                                                                       \
    float *p = buf;                                                          \
    double t0 = now();                                                       \
    __asm__ volatile(                                                        \
        "movi v16.16b,#0x3c\n\t" "movi v17.16b,#0x3c\n\t"                    \
        "movi v0.16b,#0\n\tmovi v1.16b,#0\n\tmovi v2.16b,#0\n\t"             \
        "movi v3.16b,#0\n\tmovi v4.16b,#0\n\tmovi v5.16b,#0\n\t"             \
        "movi v6.16b,#0\n\tmovi v7.16b,#0\n\t"                               \
        "mov x2,%1\n"                                                        \
        "1:\n\t" INSN " v0" SUF ", v16" SUF ", v17" SUF "\n\t"                \
        INSN " v1" SUF ", v16" SUF ", v17" SUF "\n\t"                        \
        INSN " v2" SUF ", v16" SUF ", v17" SUF "\n\t"                        \
        INSN " v3" SUF ", v16" SUF ", v17" SUF "\n\t"                        \
        INSN " v4" SUF ", v16" SUF ", v17" SUF "\n\t"                        \
        INSN " v5" SUF ", v16" SUF ", v17" SUF "\n\t"                        \
        INSN " v6" SUF ", v16" SUF ", v17" SUF "\n\t"                        \
        INSN " v7" SUF ", v16" SUF ", v17" SUF "\n\t"                        \
        "subs x2,x2,#1\n\tb.ne 1b\n\t"                                       \
        "stp q0,q1,[%0]\n\tstp q2,q3,[%0,#32]\n\t"                           \
        : "+r"(p)                                                            \
        : "r"(IT)                                                            \
        : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "x2",\
          "cc", "memory");                                                   \
    printf("%-20s %5.2f cycles / 8 instructions\n", NAME,                    \
           (now() - t0) * 3.26e9 / IT);                                      \
  } while (0)

int main(void) {
  RUN("fmla v.4s (fp32)", "fmla", ".4s");
  RUN("fmla v.8h (fp16)", "fmla", ".8h");
  RUN("fmul v.8h (fp16)", "fmul", ".8h");
  RUN("fadd v.8h (fp16)", "fadd", ".8h");
  RUN("fmaxnm v.8h", "fmaxnm", ".8h");
  return buf[0] == 0;
}
