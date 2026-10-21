// Clean peak-throughput probes for the m9g (Graviton 5 / Neoverse-V3).
//
//   gcc -O3 -march=armv8.2-a+bf16+fp16 -o one_fma one_fma.c && ./one_fma
//
// The inner loops are pure asm: 8 independent accumulators in v0-v7, invariant
// operands in v16/v17 initialised inside the asm, and every accumulator stored
// to a buffer at the end. No compiler-visible operands means no DCE (which cost
// the intrinsics version 6 of its 8 accumulators) and no register moves.
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}

static float buf[64];
static long ITERS = 2000000;

#define LOOP8(insn)                                                     \
  "mov x2, %1\n"                                                        \
  "1:\n\t" insn " v0.4s, v16.4s, v17.4s\n\t"                            \
  insn " v1.4s, v16.4s, v17.4s\n\t" insn " v2.4s, v16.4s, v17.4s\n\t"   \
  insn " v3.4s, v16.4s, v17.4s\n\t" insn " v4.4s, v16.4s, v17.4s\n\t"   \
  insn " v5.4s, v16.4s, v17.4s\n\t" insn " v6.4s, v16.4s, v17.4s\n\t"   \
  insn " v7.4s, v16.4s, v17.4s\n\t"                                     \
  "subs x2, x2, #1\n\t"                                                 \
  "b.ne 1b\n\t"

#define LOOP8B(insn)                                                    \
  "mov x2, %1\n"                                                        \
  "1:\n\t" insn " v0.4s, v16.8h, v17.8h\n\t"                            \
  insn " v1.4s, v16.8h, v17.8h\n\t" insn " v2.4s, v16.8h, v17.8h\n\t"   \
  insn " v3.4s, v16.8h, v17.8h\n\t" insn " v4.4s, v16.8h, v17.8h\n\t"   \
  insn " v5.4s, v16.8h, v17.8h\n\t" insn " v6.4s, v16.8h, v17.8h\n\t"   \
  insn " v7.4s, v16.8h, v17.8h\n\t"                                     \
  "subs x2, x2, #1\n\t"                                                 \
  "b.ne 1b\n\t"

static double run(const char *name, double flops_per_iter, const char *zero8,
                  const char *const, char width) {
  float *p = buf;
  double t0 = now();
  if (width == 'f') {
    __asm__ volatile(
        "fmov v16.4s, #1.0\n\t"
        "movi v0.16b, #0\n\tmovi v1.16b, #0\n\tmovi v2.16b, #0\n\t"
        "movi v3.16b, #0\n\tmovi v4.16b, #0\n\tmovi v5.16b, #0\n\t"
        "movi v6.16b, #0\n\tmovi v7.16b, #0\n\t"
        "mov v17.16b, v16.16b\n\t"
        LOOP8("fmla") "stp q0, q1, [%0]\n\tstp q2, q3, [%0, #32]\n\t"
        "stp q4, q5, [%0, #64]\n\tstp q6, q7, [%0, #96]\n\t"
        : "+r"(p)
        : "r"(ITERS)
        : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "x2",
          "cc", "memory");
  } else {
    __asm__ volatile(
        "movi v16.16b, #0x3f\n\t"
        "movi v0.16b, #0\n\tmovi v1.16b, #0\n\tmovi v2.16b, #0\n\t"
        "movi v3.16b, #0\n\tmovi v4.16b, #0\n\tmovi v5.16b, #0\n\t"
        "movi v6.16b, #0\n\tmovi v7.16b, #0\n\t"
        "mov v17.16b, v16.16b\n\t"
        LOOP8B("bfmmla") "stp q0, q1, [%0]\n\tstp q2, q3, [%0, #32]\n\t"
        "stp q4, q5, [%0, #64]\n\tstp q6, q7, [%0, #96]\n\t"
        : "+r"(p)
        : "r"(ITERS)
        : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "x2",
          "cc", "memory");
  }
  double dt = now() - t0;
  double g = (double)ITERS * flops_per_iter / dt / 1e9;
  printf("  %-14s %7.1f GFLOP/s  (%.2f instr/cycle at 3.26 GHz)  sink=%g\n",
         name, g, g / 3.26e9 / (flops_per_iter / 8), buf[0] + buf[7]);
  return g;
}

int main(int argc, char **argv) {
  if (argc > 1) ITERS = atol(argv[1]);
  printf("iters=%ld\n", ITERS);
  run("fp32 fmla 4s", 8.0 * 4 * 2, 0, 0, 'f');
  run("bf16 bfmmla", 8.0 * 32, 0, 0, 'b');
  return 0;
}
