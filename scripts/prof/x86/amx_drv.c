// Standalone driver for dense_amx.c so ASan/gdb can see it: runs the debug QK
// probe, the full kernel and the pack self-test against scalar references.
//   gcc -O1 -g -fsanitize=address -mamx-tile -mamx-bf16 -mavx512bf16 ... \
//       -o /tmp/drv scripts/prof/x86/amx_drv.c cpu_attention/_csrc/dense_amx.c
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int dense_amx_bf16(const void *q, const void *k, const void *v, void *out,
                   long batch, long seq_q, long seq_k, long nh, long nkv,
                   long head_dim, long qb, long qt, long qh, long ksb, long kst,
                   long ksh, long vsb, long vst, long vsh, long ob, long ot,
                   long oh, double scale, long causal, long nthreads, long kt);
int dense_amx_ok(void);
int dense_amx_debug_qk(const void *q, const void *k, long seq_k, long D,
                       void *logits);
int dense_amx_pack_selftest(void);

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

int main(void) {
  printf("amx ok %d, pack selftest %d\n", dense_amx_ok(),
         dense_amx_pack_selftest());
  fflush(stdout);
  srand(1);
  for (int D = 32; D <= 128; D *= 2) {
    enum { SQ = 64, SK = 64, NH = 2 };
    enum { MAXN = 64 * 128 };
    static float q[MAXN], k[MAXN], v[MAXN];
    static unsigned short qb[MAXN], kb[MAXN], vb[MAXN], ob[MAXN];
    for (int i = 0; i < SQ * D; i++) q[i] = (rand() % 200 - 100) / 50.0f;
    for (int i = 0; i < SK * D; i++) {
      k[i] = (rand() % 200 - 100) / 50.0f;
      v[i] = (rand() % 200 - 100) / 50.0f;
    }
    for (int i = 0; i < SQ * D; i++) qb[i] = f2bf(q[i]);
    for (int i = 0; i < SK * D; i++) {
      kb[i] = f2bf(k[i]);
      vb[i] = f2bf(v[i]);
    }
    /* ---- QK probe ---- */
    static float lg[16 * 32];
    int rc = dense_amx_debug_qk(qb, kb, SK, D, lg);
    float worst = 0;
    for (int m = 0; m < 16; m++)
      for (int n = 0; n < 32; n++) {
        double s = 0;
        for (int d = 0; d < D; d++)
          s += (double)bf2f(qb[m * D + d]) * bf2f(kb[n * D + d]);
        float e = fabsf((float)s - lg[m * 32 + n]);
        if (e > worst) worst = e;
      }
    printf("D=%d QK probe rc=%d max|d|=%g\n", D, rc, worst);
    fflush(stdout);

    /* ---- whole kernel vs a scalar reference ---- */
    const double scale = 1.0 / sqrt((double)D);
    rc = dense_amx_bf16(qb, kb, vb, ob, 1, SQ, SK, 1, 1, D, SQ * D, D, 1,
                        SK * D, D, 1, SK * D, D, 1, SQ * D, D, 1, scale, 1, 1,
                        128);
    worst = 0;
    for (int n = 0; n < SQ; n++) {
      double mx = -1e30;
      static double p[SK];
      for (int j = 0; j <= n; j++) {
        double s = 0;
        for (int d = 0; d < D; d++)
          s += (double)bf2f(qb[n * D + d]) * bf2f(kb[j * D + d]);
        p[j] = s * scale;
        if (p[j] > mx) mx = p[j];
      }
      double sum = 0;
      for (int j = 0; j <= n; j++) {
        p[j] = exp(p[j] - mx);
        sum += p[j];
      }
      for (int d = 0; d < D; d++) {
        double o = 0;
        for (int j = 0; j <= n; j++) o += p[j] * bf2f(vb[j * D + d]);
        o /= sum;
        float e = fabsf((float)o - bf2f(ob[n * D + d]));
        if (e > worst) worst = e;
      }
    }
    printf("D=%d kernel rc=%d max|d| vs scalar=%g\n", D, rc, worst);
    fflush(stdout);
    (void)NH;
  }
  return 0;
}
