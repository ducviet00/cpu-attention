// Probe the V (PV B-operand) key mapping with one-hot V columns.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int dense_amx_bf16(const void *q, const void *k, const void *v, void *out,
                   long batch, long seq_q, long seq_k, long nh, long nkv,
                   long head_dim, long qb, long qt, long qh, long ksb, long kst,
                   long ksh, long vsb, long vst, long vsh, long ob, long ot,
                   long oh, double scale, long causal, long nthreads, long kt);

static unsigned short f2bf(float f) {
  unsigned u;
  memcpy(&u, &f, 4);
  u = (u + 0x7fff + ((u >> 16) & 1)) >> 16;
  return (unsigned short)u;
}
static float bf2f(unsigned short b) {
  unsigned u = (unsigned)b << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}
enum { SQ = 32, SK = 32, MAXD = 128 };
static unsigned short q[MAXD * SQ], k[MAXD * SK], v[MAXD * SK], o[MAXD * SQ];

int main(int argc, char **argv) {
  const int D = argc > 1 ? atoi(argv[1]) : 32;
  const int nk = argc > 2 ? atoi(argv[2]) : SK;
  const double sc = 1.0 / sqrt((double)D);
  for (int i = 0; i < D * SQ; i++) q[i] = f2bf(0.5f);
  for (int i = 0; i < D * SK; i++) k[i] = f2bf(0.5f);
  // uniform logits -> all weights equal; V one-hot in the key index -> the
  // output value for dim 0 across j shows how key j is weighted.
  printf("D=%d nk=%d: for each key j, out[31][0] with V=onehot(j) (expect 1/nk = %g)\n",
         D, nk, 1.0 / nk);
  for (int j0 = 0; j0 < nk; j0++) {
    memset(v, 0, sizeof v);
    for (int d = 0; d < D; d++) v[j0 * D + d] = f2bf(1.0f);
    dense_amx_bf16(q, k, v, o, 1, SQ, nk, 1, 1, D, SQ * D, D, 1, nk * D, D, 1,
                   nk * D, D, 1, SQ * D, D, 1, sc, 0, 1, 128);
    printf("  j=%2d -> %g", j0, bf2f(o[31 * D]));
    if (j0 % 4 == 3) printf("\n");
  }
  printf("\n");
  return 0;
}
