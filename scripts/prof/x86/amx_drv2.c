// Focused AMX kernel debugging: probe the kernel with inputs whose correct
// output is known analytically, so the failing phase is obvious.
//   gcc -O1 -g -fopenmp -fsanitize=address -mavx512f -mavx512bw -mavx512dq \
//       -mavx512vl -mavx512bf16 -mamx-tile -mamx-bf16 -o /tmp/drv2 \
//       /tmp/amx_drv2.c dense_amx.c -lm && /tmp/drv2
#include <math.h>
#include <stdio.h>
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

static void run(int D, int causal, double scale) {
  int rc = dense_amx_bf16(q, k, v, o, 1, SQ, SK, 1, 1, D, SQ * D, D, 1, SK * D,
                          D, 1, SK * D, D, 1, SQ * D, D, 1, scale, causal, 1,
                          128);
  printf("  rc=%d out[0][0..5]:", rc);
  for (int j = 0; j < 6; j++) printf(" %g", bf2f(o[j]));
  printf("\n  out[17][0..5]:");
  for (int j = 0; j < 6; j++) printf(" %g", bf2f(o[17 * D + j]));
  printf("\n  out[31][0..5]:");
  for (int j = 0; j < 6; j++) printf(" %g", bf2f(o[31 * D + j]));
  printf("\n");
  fflush(stdout);
}

int main(void) {
  const int D = 32;
  const double sc = 1.0 / sqrt((double)D);
  /* 1. V = 1 everywhere -> output must be exactly 1 */
  for (int i = 0; i < D * SQ; i++) q[i] = f2bf(0.5f);
  for (int i = 0; i < D * SK; i++) {
    k[i] = f2bf(0.5f);
    v[i] = f2bf(1.0f);
  }
  printf("1. V=1 (expect all 1.0), causal=0\n");
  run(D, 0, sc);
  /* 2. V[j][d] = j -> output = sum_j p_j * j / sum_j p_j, uniform logits */
  for (int j = 0; j < SK; j++)
    for (int d = 0; d < D; d++) v[j * D + d] = f2bf((float)j);
  printf("2. V[j]=j, uniform logits (expect 15.5 everywhere), causal=0\n");
  run(D, 0, sc);
  /* 3. same but causal: row r sees keys 0..r -> expect r/2 */
  printf("3. V[j]=j, causal (row r expects r/2: 0, 8.5, 15.5)\n");
  run(D, 1, sc);
  /* 4. V = identity in d -> out[r][d] = p_d/sum */
  for (int i = 0; i < D * SK; i++) v[i] = 0;
  for (int j = 0; j < SK && j < D; j++) v[j * D + j] = f2bf(1.0f);
  printf("4. V=identity, uniform logits (expect 1/32 each), causal=0\n");
  run(D, 0, sc);
  return 0;
}
