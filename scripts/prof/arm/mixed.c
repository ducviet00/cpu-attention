// Do bfmmla and fp32 fmla co-issue? Neoverse-V3 has 4 SIMD pipes and bfmmla
// only sustains 2/cycle, so if fmla uses the other two a mixed kernel gets up to
// 32 + 13 = 45 MACs/cycle instead of 32.
//   gcc -O3 -march=armv8.2-a+bf16+fp16 -o mixed mixed.c && ./mixed
#include <stdio.h>
#include <time.h>

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}

static float buf[128];
static long IT = 2000000;

#define HEAD(n)                                                              \
  "movi v16.16b,#0x3f\n\t" /* bf16 1.0 x8 */                                 \
  "movi v17.16b,#0x3f\n\t"                                                   \
  "fmov v18.4s,#1.0\n\t"                                                     \
  "movi v0.16b,#0\n\tmovi v1.16b,#0\n\tmovi v2.16b,#0\n\tmovi v3.16b,#0\n\t" \
  "movi v4.16b,#0\n\tmovi v5.16b,#0\n\tmovi v6.16b,#0\n\tmovi v7.16b,#0\n\t" \
  "movi v8.16b,#0\n\tmovi v9.16b,#0\n\tmovi v10.16b,#0\n\tmovi v11.16b,#0\n\t" \
  "movi v12.16b,#0\n\tmovi v13.16b,#0\n\tmovi v14.16b,#0\n\tmovi v15.16b,#0\n\t" \
  "mov x2,%1\n"                                                              \
  "1:\n\t"

#define TAIL                                                                 \
  "subs x2,x2,#1\n\tb.ne 1b\n\t"                                             \
  "stp q0,q1,[%0]\n\tstp q2,q3,[%0,#32]\n\tstp q4,q5,[%0,#64]\n\t"           \
  "stp q6,q7,[%0,#96]\n\t"                                                   \
  : "+r"(p)                                                                  \
  : "r"(IT)                                                                  \
  : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",       \
    "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "x2", "cc",      \
    "memory"

#define B(i) "bfmmla v" #i ".4s, v16.8h, v17.8h\n\t"
#define F(i) "fmla v" #i ".4s, v18.4s, v18.4s\n\t"

int main(void) {
  double t0;
  float *p = buf;
  // 8 bfmmla alone
  t0 = now();
  __asm__ volatile(HEAD("") B(0) B(1) B(2) B(3) B(4) B(5) B(6) B(7) TAIL);
  printf("8 bfmmla            %5.2f cyc/iter\n", (now() - t0) * 3.26e9 / IT);
  // 8 fmla alone
  t0 = now();
  __asm__ volatile(HEAD("") F(0) F(1) F(2) F(3) F(4) F(5) F(6) F(7) TAIL);
  printf("8 fmla              %5.2f cyc/iter\n", (now() - t0) * 3.26e9 / IT);
  // 8 + 8 mixed: if they co-issue this should be ~max(4, 2.4) = 4, else 6.4
  t0 = now();
  __asm__ volatile(HEAD("") B(0) F(8) B(1) F(9) B(2) F(10) B(3) F(11) B(4)
                       F(12) B(5) F(13) B(6) F(14) B(7) F(15) TAIL);
  printf("8 bfmmla + 8 fmla   %5.2f cyc/iter  (co-issue if ~4.0)\n",
         (now() - t0) * 3.26e9 / IT);
  return 0;
}
