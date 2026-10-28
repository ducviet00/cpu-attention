# Per-row Axion results

Each ratio uses arms timed in the same process. No cross-run speedups.

## `paged-full-afbea06.json`

Rows: 92; {'loss': 55, 'unresolved': 24, 'win': 11, 'no valid baseline': 2}.
Invalid new outputs: [].

| Row | Verdict | New / valid C++ | New / previous | Control spread |
|---|---|---:|---:|---:|
| paged engine self b=1 ctx=256 bf16 | loss | 2.253 | — | 0.026 |
| paged engine self b=1 ctx=256 fp32 | loss | 1.409 | — | 0.069 |
| paged engine cross b=1 ctx=256 bf16 | loss | 2.295 | — | 0.024 |
| paged engine cross b=1 ctx=256 fp32 | loss | 1.421 | — | 0.030 |
| paged engine self b=8 ctx=256 bf16 | loss | 1.882 | — | 0.015 |
| paged engine self b=8 ctx=256 fp32 | loss | 1.082 | — | 0.048 |
| paged engine cross b=8 ctx=256 bf16 | loss | 1.849 | — | 0.018 |
| paged engine cross b=8 ctx=256 fp32 | unresolved | 0.976 | — | 0.012 |
| paged engine self b=8 ctx=1024 bf16 | loss | 1.553 | — | 0.004 |
| paged engine self b=8 ctx=1024 fp32 | win | 0.924 | — | 0.044 |
| paged engine cross b=8 ctx=1024 bf16 | loss | 1.654 | — | 0.012 |
| paged engine cross b=8 ctx=1024 fp32 | unresolved | 0.983 | — | 0.015 |
| paged engine self b=32 ctx=1024 bf16 | loss | 1.323 | — | 0.048 |
| paged engine self b=32 ctx=1024 fp32 | unresolved | 1.005 | — | 0.052 |
| paged engine cross b=32 ctx=1024 bf16 | loss | 1.371 | — | 0.006 |
| paged engine cross b=32 ctx=1024 fp32 | unresolved | 0.990 | — | 0.013 |
| paged engine self b=1 ctx=4096 bf16 | loss | 1.378 | — | 0.018 |
| paged engine self b=1 ctx=4096 fp32 | unresolved | 1.025 | — | 0.030 |
| paged engine cross b=1 ctx=4096 bf16 | loss | 1.419 | — | 0.017 |
| paged engine cross b=1 ctx=4096 fp32 | unresolved | 0.990 | — | 0.018 |
| paged engine self b=8 ctx=255 bf16 | loss | 1.896 | — | 0.012 |
| paged engine self b=8 ctx=255 fp32 | unresolved | 1.030 | — | 0.066 |
| paged engine cross b=8 ctx=255 bf16 | loss | 1.834 | — | 0.019 |
| paged engine cross b=8 ctx=255 fp32 | unresolved | 1.011 | — | 0.041 |
| paged engine self b=8 ctx=257 bf16 | loss | 1.619 | — | 0.022 |
| paged engine self b=8 ctx=257 fp32 | unresolved | 0.991 | — | 0.035 |
| paged engine cross b=8 ctx=257 bf16 | loss | 1.611 | — | 0.004 |
| paged engine cross b=8 ctx=257 fp32 | loss | 1.031 | — | 0.003 |
| paged engine self b=8 ctx=1023 bf16 | loss | 1.608 | — | 0.016 |
| paged engine self b=8 ctx=1023 fp32 | win | 0.959 | — | 0.017 |
| paged engine cross b=8 ctx=1023 bf16 | loss | 1.586 | — | 0.023 |
| paged engine cross b=8 ctx=1023 fp32 | win | 0.966 | — | 0.004 |
| paged engine self b=8 ctx=1025 bf16 | loss | 1.576 | — | 0.021 |
| paged engine self b=8 ctx=1025 fp32 | win | 0.944 | — | 0.023 |
| paged engine cross b=8 ctx=1025 bf16 | loss | 1.579 | — | 0.028 |
| paged engine cross b=8 ctx=1025 fp32 | unresolved | 0.983 | — | 0.018 |
| paged engine self b=1 ctx=4097 bf16 | loss | 1.408 | — | 0.041 |
| paged engine self b=1 ctx=4097 fp32 | loss | 1.041 | — | 0.038 |
| paged engine cross b=1 ctx=4097 bf16 | loss | 1.400 | — | 0.011 |
| paged engine cross b=1 ctx=4097 fp32 | loss | 1.044 | — | 0.038 |
| paged engine self b=8 ctx=1024 page=16 bf16 | loss | 1.575 | — | 0.018 |
| paged engine self b=8 ctx=1024 page=16 fp32 | win | 0.949 | — | 0.015 |
| paged engine self b=8 ctx=1024 page=48 bf16 | loss | 1.960 | — | 0.020 |
| paged engine self b=8 ctx=1024 page=48 fp32 | win | 0.940 | — | 0.021 |
| paged engine self b=8 ctx=1024 page=64 bf16 | loss | 1.731 | — | 0.000 |
| paged engine self b=8 ctx=1024 page=64 fp32 | unresolved | 1.014 | — | 0.020 |
| paged engine self b=8 ctx=1024 h=16/4 bf16 | loss | 1.601 | — | 0.043 |
| paged engine self b=8 ctx=1024 h=16/4 fp32 | win | 0.714 | — | 0.049 |
| paged engine cross b=8 ctx=1024 h=16/4 bf16 | loss | 1.648 | — | 0.032 |
| paged engine cross b=8 ctx=1024 h=16/4 fp32 | win | 0.721 | — | 0.086 |
| paged engine cross b=8 ctx=1024 h=16/1 bf16 | loss | 2.047 | — | 0.011 |
| paged engine cross b=8 ctx=1024 h=16/1 fp32 | unresolved | 0.978 | — | 0.003 |
| paged engine self b=8 ctx=het bf16 | no valid baseline | — | — | — |
| paged engine self b=8 ctx=het fp32 | win | 0.958 | — | 0.024 |
| paged engine cross b=8 ctx=het bf16 | no valid baseline | — | — | — |
| paged engine cross b=8 ctx=het fp32 | unresolved | 0.965 | — | 0.040 |
| paged engine cross b=8 ctx=1024 d=32 bf16 | loss | 1.696 | — | 0.016 |
| paged engine cross b=8 ctx=1024 d=32 fp32 | unresolved | 1.022 | — | 0.019 |
| paged engine cross b=8 ctx=1024 d=128 bf16 | loss | 1.920 | — | 0.035 |
| paged engine cross b=8 ctx=1024 d=128 fp32 | loss | 1.052 | — | 0.041 |
| paged store store b=32 ctx=1 bf16 | win | 0.965 | — | 0.018 |
| paged store store b=32 ctx=1 fp32 | loss | 1.036 | — | 0.010 |
| paged store store b=1 ctx=1024 bf16 | win | 0.903 | — | 0.009 |
| paged store store b=1 ctx=1024 fp32 | loss | 1.048 | — | 0.011 |
| paged engine_out self b=1 ctx=256 bf16 | loss | 2.346 | — | 0.053 |
| paged engine_out self b=1 ctx=256 fp32 | loss | 1.136 | — | 0.053 |
| paged engine_out cross b=1 ctx=256 bf16 | loss | 2.373 | — | 0.042 |
| paged engine_out cross b=1 ctx=256 fp32 | loss | 1.445 | — | 0.200 |
| paged engine_out self b=8 ctx=256 bf16 | loss | 1.851 | — | 0.002 |
| paged engine_out self b=8 ctx=256 fp32 | unresolved | 0.971 | — | 0.006 |
| paged engine_out cross b=8 ctx=256 bf16 | loss | 1.792 | — | 0.002 |
| paged engine_out cross b=8 ctx=256 fp32 | unresolved | 0.928 | — | 0.119 |
| paged engine_out self b=8 ctx=1024 bf16 | loss | 1.539 | — | 0.019 |
| paged engine_out self b=8 ctx=1024 fp32 | unresolved | 0.976 | — | 0.011 |
| paged engine_out cross b=8 ctx=1024 bf16 | loss | 1.565 | — | 0.025 |
| paged engine_out cross b=8 ctx=1024 fp32 | unresolved | 0.981 | — | 0.004 |
| paged engine_out self b=32 ctx=1024 bf16 | loss | 1.343 | — | 0.007 |
| paged engine_out self b=32 ctx=1024 fp32 | unresolved | 0.999 | — | 0.008 |
| paged engine_out cross b=32 ctx=1024 bf16 | loss | 1.288 | — | 0.018 |
| paged engine_out cross b=32 ctx=1024 fp32 | unresolved | 0.982 | — | 0.008 |
| paged engine_out self b=1 ctx=4096 bf16 | loss | 1.384 | — | 0.004 |
| paged engine_out self b=1 ctx=4096 fp32 | unresolved | 0.989 | — | 0.033 |
| paged engine_out cross b=1 ctx=4096 bf16 | loss | 1.406 | — | 0.021 |
| paged engine_out cross b=1 ctx=4096 fp32 | unresolved | 0.985 | — | 0.030 |
| paged api self b=1 ctx=256 bf16 | loss | 1.150 | — | 0.001 |
| paged api self b=1 ctx=256 fp32 | loss | 1.105 | — | 0.012 |
| paged api cross b=1 ctx=256 bf16 | loss | 1.623 | — | 0.004 |
| paged api cross b=1 ctx=256 fp32 | loss | 1.219 | — | 0.028 |
| paged api self b=8 ctx=1024 bf16 | loss | 1.407 | — | 0.017 |
| paged api self b=8 ctx=1024 fp32 | unresolved | 1.014 | — | 0.030 |
| paged api cross b=8 ctx=1024 bf16 | loss | 1.513 | — | 0.022 |
| paged api cross b=8 ctx=1024 fp32 | unresolved | 0.990 | — | 0.027 |

## `paged-full-bfdot.json`

Rows: 92; {'loss': 51, 'unresolved': 23, 'win': 16, 'INVALID': 2}.
Invalid new outputs: ['paged engine self b=8 ctx=het bf16', 'paged engine cross b=8 ctx=het bf16'].

| Row | Verdict | New / valid C++ | New / previous | Control spread |
|---|---|---:|---:|---:|
| paged engine self b=1 ctx=256 bf16 | loss | 1.624 | 0.736 | 0.118 |
| paged engine self b=1 ctx=256 fp32 | loss | 1.531 | 1.179 | 0.072 |
| paged engine cross b=1 ctx=256 bf16 | loss | 1.628 | 0.725 | 0.023 |
| paged engine cross b=1 ctx=256 fp32 | loss | 1.531 | 1.137 | 0.088 |
| paged engine self b=8 ctx=256 bf16 | loss | 1.137 | 0.639 | 0.107 |
| paged engine self b=8 ctx=256 fp32 | loss | 1.054 | 1.042 | 0.036 |
| paged engine cross b=8 ctx=256 bf16 | loss | 1.035 | 0.630 | 0.033 |
| paged engine cross b=8 ctx=256 fp32 | unresolved | 1.015 | 0.970 | 0.009 |
| paged engine self b=8 ctx=1024 bf16 | loss | 1.130 | 0.762 | 0.003 |
| paged engine self b=8 ctx=1024 fp32 | win | 0.906 | 0.978 | 0.023 |
| paged engine cross b=8 ctx=1024 bf16 | loss | 1.149 | 0.737 | 0.010 |
| paged engine cross b=8 ctx=1024 fp32 | win | 0.942 | 0.996 | 0.020 |
| paged engine self b=32 ctx=1024 bf16 | loss | 1.127 | 0.815 | 0.043 |
| paged engine self b=32 ctx=1024 fp32 | unresolved | 1.008 | 0.994 | 0.032 |
| paged engine cross b=32 ctx=1024 bf16 | loss | 1.086 | 0.804 | 0.000 |
| paged engine cross b=32 ctx=1024 fp32 | loss | 1.061 | 1.006 | 0.021 |
| paged engine self b=1 ctx=4096 bf16 | loss | 1.031 | 0.710 | 0.023 |
| paged engine self b=1 ctx=4096 fp32 | unresolved | 1.018 | 1.041 | 0.012 |
| paged engine cross b=1 ctx=4096 bf16 | loss | 1.065 | 0.722 | 0.029 |
| paged engine cross b=1 ctx=4096 fp32 | unresolved | 1.007 | 1.024 | 0.004 |
| paged engine self b=8 ctx=255 bf16 | unresolved | 1.054 | 0.581 | 0.062 |
| paged engine self b=8 ctx=255 fp32 | loss | 1.070 | 1.044 | 0.036 |
| paged engine cross b=8 ctx=255 bf16 | unresolved | 1.055 | 0.616 | 0.072 |
| paged engine cross b=8 ctx=255 fp32 | loss | 1.035 | 0.943 | 0.019 |
| paged engine self b=8 ctx=257 bf16 | unresolved | 1.079 | 0.625 | 0.094 |
| paged engine self b=8 ctx=257 fp32 | loss | 1.109 | 1.139 | 0.101 |
| paged engine cross b=8 ctx=257 bf16 | unresolved | 1.038 | 0.628 | 0.047 |
| paged engine cross b=8 ctx=257 fp32 | unresolved | 1.086 | 0.982 | 0.096 |
| paged engine self b=8 ctx=1023 bf16 | loss | 1.244 | 0.791 | 0.047 |
| paged engine self b=8 ctx=1023 fp32 | win | 0.963 | 0.989 | 0.016 |
| paged engine cross b=8 ctx=1023 bf16 | loss | 1.097 | 0.726 | 0.034 |
| paged engine cross b=8 ctx=1023 fp32 | unresolved | 0.986 | 0.996 | 0.040 |
| paged engine self b=8 ctx=1025 bf16 | loss | 1.108 | 0.724 | 0.010 |
| paged engine self b=8 ctx=1025 fp32 | win | 0.959 | 0.990 | 0.013 |
| paged engine cross b=8 ctx=1025 bf16 | loss | 1.129 | 0.728 | 0.003 |
| paged engine cross b=8 ctx=1025 fp32 | win | 0.967 | 1.004 | 0.026 |
| paged engine self b=1 ctx=4097 bf16 | loss | 1.082 | 0.715 | 0.052 |
| paged engine self b=1 ctx=4097 fp32 | unresolved | 1.006 | 1.014 | 0.001 |
| paged engine cross b=1 ctx=4097 bf16 | loss | 1.097 | 0.704 | 0.058 |
| paged engine cross b=1 ctx=4097 fp32 | unresolved | 1.016 | 1.041 | 0.007 |
| paged engine self b=8 ctx=1024 page=16 bf16 | loss | 1.078 | 0.729 | 0.007 |
| paged engine self b=8 ctx=1024 page=16 fp32 | win | 0.921 | 1.019 | 0.012 |
| paged engine self b=8 ctx=1024 page=48 bf16 | loss | 1.109 | 0.653 | 0.014 |
| paged engine self b=8 ctx=1024 page=48 fp32 | win | 0.940 | 0.991 | 0.002 |
| paged engine self b=8 ctx=1024 page=64 bf16 | loss | 1.065 | 0.644 | 0.002 |
| paged engine self b=8 ctx=1024 page=64 fp32 | unresolved | 1.026 | 1.018 | 0.046 |
| paged engine self b=8 ctx=1024 h=16/4 bf16 | unresolved | 1.002 | 0.648 | 0.070 |
| paged engine self b=8 ctx=1024 h=16/4 fp32 | win | 0.757 | 1.014 | 0.055 |
| paged engine cross b=8 ctx=1024 h=16/4 bf16 | unresolved | 0.998 | 0.647 | 0.071 |
| paged engine cross b=8 ctx=1024 h=16/4 fp32 | win | 0.860 | 1.054 | 0.058 |
| paged engine cross b=8 ctx=1024 h=16/1 bf16 | loss | 1.213 | 0.607 | 0.024 |
| paged engine cross b=8 ctx=1024 h=16/1 fp32 | win | 0.939 | 1.022 | 0.021 |
| paged engine self b=8 ctx=het bf16 | INVALID | 0.177 | 0.756 | 0.040 |
| paged engine self b=8 ctx=het fp32 | unresolved | 0.985 | 1.022 | 0.005 |
| paged engine cross b=8 ctx=het bf16 | INVALID | 0.184 | 0.748 | 0.038 |
| paged engine cross b=8 ctx=het fp32 | win | 0.968 | 1.006 | 0.003 |
| paged engine cross b=8 ctx=1024 d=32 bf16 | loss | 1.160 | 0.695 | 0.003 |
| paged engine cross b=8 ctx=1024 d=32 fp32 | loss | 1.045 | 1.035 | 0.037 |
| paged engine cross b=8 ctx=1024 d=128 bf16 | loss | 1.312 | 0.696 | 0.033 |
| paged engine cross b=8 ctx=1024 d=128 fp32 | unresolved | 0.981 | 1.039 | 0.022 |
| paged store store b=32 ctx=1 bf16 | loss | 1.125 | 1.055 | 0.015 |
| paged store store b=32 ctx=1 fp32 | loss | 1.129 | 1.039 | 0.062 |
| paged store store b=1 ctx=1024 bf16 | win | 0.845 | 1.006 | 0.005 |
| paged store store b=1 ctx=1024 fp32 | loss | 1.055 | 1.018 | 0.006 |
| paged engine_out self b=1 ctx=256 bf16 | loss | 1.454 | 0.684 | 0.079 |
| paged engine_out self b=1 ctx=256 fp32 | loss | 1.322 | 1.106 | 0.022 |
| paged engine_out cross b=1 ctx=256 bf16 | loss | 1.499 | 0.655 | 0.049 |
| paged engine_out cross b=1 ctx=256 fp32 | loss | 1.299 | 1.078 | 0.048 |
| paged engine_out self b=8 ctx=256 bf16 | loss | 1.035 | 0.581 | 0.021 |
| paged engine_out self b=8 ctx=256 fp32 | loss | 1.069 | 1.060 | 0.037 |
| paged engine_out cross b=8 ctx=256 bf16 | unresolved | 0.971 | 0.575 | 0.040 |
| paged engine_out cross b=8 ctx=256 fp32 | win | 0.962 | 0.954 | 0.023 |
| paged engine_out self b=8 ctx=1024 bf16 | loss | 1.120 | 0.728 | 0.002 |
| paged engine_out self b=8 ctx=1024 fp32 | win | 0.928 | 1.020 | 0.009 |
| paged engine_out cross b=8 ctx=1024 bf16 | loss | 1.191 | 0.732 | 0.022 |
| paged engine_out cross b=8 ctx=1024 fp32 | win | 0.935 | 1.019 | 0.000 |
| paged engine_out self b=32 ctx=1024 bf16 | loss | 1.055 | 0.799 | 0.004 |
| paged engine_out self b=32 ctx=1024 fp32 | win | 0.963 | 0.965 | 0.032 |
| paged engine_out cross b=32 ctx=1024 bf16 | loss | 1.098 | 0.813 | 0.010 |
| paged engine_out cross b=32 ctx=1024 fp32 | unresolved | 1.003 | 0.992 | 0.033 |
| paged engine_out self b=1 ctx=4096 bf16 | loss | 1.037 | 0.690 | 0.006 |
| paged engine_out self b=1 ctx=4096 fp32 | unresolved | 0.983 | 1.027 | 0.004 |
| paged engine_out cross b=1 ctx=4096 bf16 | loss | 1.063 | 0.693 | 0.009 |
| paged engine_out cross b=1 ctx=4096 fp32 | unresolved | 0.971 | 1.024 | 0.007 |
| paged api self b=1 ctx=256 bf16 | loss | 1.070 | 0.920 | 0.009 |
| paged api self b=1 ctx=256 fp32 | loss | 1.050 | 1.001 | 0.019 |
| paged api cross b=1 ctx=256 bf16 | loss | 1.153 | 0.793 | 0.013 |
| paged api cross b=1 ctx=256 fp32 | loss | 1.160 | 1.015 | 0.023 |
| paged api self b=8 ctx=1024 bf16 | loss | 1.100 | 0.780 | 0.073 |
| paged api self b=8 ctx=1024 fp32 | unresolved | 1.026 | 1.000 | 0.072 |
| paged api cross b=8 ctx=1024 bf16 | loss | 1.082 | 0.720 | 0.041 |
| paged api cross b=8 ctx=1024 fp32 | unresolved | 1.021 | 1.006 | 0.068 |

## `paged-full-bfdot-hilo.json`

Rows: 92; {'loss': 46, 'unresolved': 30, 'win': 15, 'no valid baseline': 1}.
Invalid new outputs: ['paged store store b=32 ctx=1 bf16'].

| Row | Verdict | New / valid C++ | New / previous | Control spread |
|---|---|---:|---:|---:|
| paged engine self b=1 ctx=256 bf16 | loss | 1.567 | 0.733 | 0.009 |
| paged engine self b=1 ctx=256 fp32 | loss | 1.116 | 1.040 | 0.028 |
| paged engine cross b=1 ctx=256 bf16 | loss | 1.619 | 0.744 | 0.001 |
| paged engine cross b=1 ctx=256 fp32 | unresolved | 1.054 | 1.066 | 0.112 |
| paged engine self b=8 ctx=256 bf16 | loss | 1.091 | 0.617 | 0.002 |
| paged engine self b=8 ctx=256 fp32 | loss | 1.089 | 1.043 | 0.076 |
| paged engine cross b=8 ctx=256 bf16 | loss | 1.044 | 0.629 | 0.001 |
| paged engine cross b=8 ctx=256 fp32 | unresolved | 0.961 | 0.972 | 0.043 |
| paged engine self b=8 ctx=1024 bf16 | loss | 1.201 | 0.781 | 0.010 |
| paged engine self b=8 ctx=1024 fp32 | win | 0.916 | 1.018 | 0.044 |
| paged engine cross b=8 ctx=1024 bf16 | loss | 1.142 | 0.735 | 0.031 |
| paged engine cross b=8 ctx=1024 fp32 | unresolved | 0.936 | 1.033 | 0.065 |
| paged engine self b=32 ctx=1024 bf16 | loss | 1.132 | 0.844 | 0.004 |
| paged engine self b=32 ctx=1024 fp32 | unresolved | 1.014 | 1.012 | 0.018 |
| paged engine cross b=32 ctx=1024 bf16 | loss | 1.178 | 0.862 | 0.026 |
| paged engine cross b=32 ctx=1024 fp32 | unresolved | 1.019 | 0.984 | 0.006 |
| paged engine self b=1 ctx=4096 bf16 | loss | 1.083 | 0.767 | 0.023 |
| paged engine self b=1 ctx=4096 fp32 | unresolved | 0.996 | 1.005 | 0.020 |
| paged engine cross b=1 ctx=4096 bf16 | loss | 1.084 | 0.771 | 0.045 |
| paged engine cross b=1 ctx=4096 fp32 | unresolved | 1.006 | 1.011 | 0.011 |
| paged engine self b=8 ctx=255 bf16 | loss | 1.124 | 0.604 | 0.019 |
| paged engine self b=8 ctx=255 fp32 | unresolved | 1.049 | 1.074 | 0.102 |
| paged engine cross b=8 ctx=255 bf16 | loss | 1.061 | 0.628 | 0.001 |
| paged engine cross b=8 ctx=255 fp32 | unresolved | 1.052 | 1.007 | 0.127 |
| paged engine self b=8 ctx=257 bf16 | unresolved | 1.012 | 0.647 | 0.016 |
| paged engine self b=8 ctx=257 fp32 | unresolved | 1.049 | 1.068 | 0.067 |
| paged engine cross b=8 ctx=257 bf16 | unresolved | 0.980 | 0.626 | 0.018 |
| paged engine cross b=8 ctx=257 fp32 | unresolved | 1.005 | 1.070 | 0.096 |
| paged engine self b=8 ctx=1023 bf16 | loss | 1.195 | 0.786 | 0.000 |
| paged engine self b=8 ctx=1023 fp32 | win | 0.962 | 1.025 | 0.009 |
| paged engine cross b=8 ctx=1023 bf16 | loss | 1.262 | 0.803 | 0.006 |
| paged engine cross b=8 ctx=1023 fp32 | win | 0.968 | 1.022 | 0.009 |
| paged engine self b=8 ctx=1025 bf16 | loss | 1.234 | 0.805 | 0.037 |
| paged engine self b=8 ctx=1025 fp32 | win | 0.935 | 1.016 | 0.014 |
| paged engine cross b=8 ctx=1025 bf16 | loss | 1.202 | 0.824 | 0.042 |
| paged engine cross b=8 ctx=1025 fp32 | win | 0.957 | 1.024 | 0.008 |
| paged engine self b=1 ctx=4097 bf16 | loss | 1.082 | 0.739 | 0.032 |
| paged engine self b=1 ctx=4097 fp32 | unresolved | 1.021 | 1.029 | 0.000 |
| paged engine cross b=1 ctx=4097 bf16 | loss | 1.083 | 0.749 | 0.008 |
| paged engine cross b=1 ctx=4097 fp32 | loss | 1.030 | 1.043 | 0.006 |
| paged engine self b=8 ctx=1024 page=16 bf16 | loss | 1.234 | 0.786 | 0.017 |
| paged engine self b=8 ctx=1024 page=16 fp32 | unresolved | 0.971 | 1.040 | 0.020 |
| paged engine self b=8 ctx=1024 page=48 bf16 | loss | 1.177 | 0.658 | 0.026 |
| paged engine self b=8 ctx=1024 page=48 fp32 | win | 0.953 | 0.998 | 0.013 |
| paged engine self b=8 ctx=1024 page=64 bf16 | loss | 1.194 | 0.669 | 0.009 |
| paged engine self b=8 ctx=1024 page=64 fp32 | loss | 1.044 | 1.030 | 0.014 |
| paged engine self b=8 ctx=1024 h=16/4 bf16 | loss | 1.294 | 0.854 | 0.011 |
| paged engine self b=8 ctx=1024 h=16/4 fp32 | win | 0.781 | 1.050 | 0.042 |
| paged engine cross b=8 ctx=1024 h=16/4 bf16 | loss | 1.319 | 0.865 | 0.010 |
| paged engine cross b=8 ctx=1024 h=16/4 fp32 | win | 0.824 | 1.040 | 0.038 |
| paged engine cross b=8 ctx=1024 h=16/1 bf16 | loss | 1.645 | 0.822 | 0.005 |
| paged engine cross b=8 ctx=1024 h=16/1 fp32 | win | 0.963 | 0.990 | 0.027 |
| paged engine self b=8 ctx=het bf16 | win | 0.176 | 0.717 | 0.030 |
| paged engine self b=8 ctx=het fp32 | unresolved | 1.013 | 1.035 | 0.032 |
| paged engine cross b=8 ctx=het bf16 | win | 0.178 | 0.716 | 0.026 |
| paged engine cross b=8 ctx=het fp32 | unresolved | 0.979 | 1.017 | 0.010 |
| paged engine cross b=8 ctx=1024 d=32 bf16 | loss | 1.215 | 0.730 | 0.032 |
| paged engine cross b=8 ctx=1024 d=32 fp32 | loss | 1.037 | 1.013 | 0.004 |
| paged engine cross b=8 ctx=1024 d=128 bf16 | loss | 1.465 | 0.741 | 0.007 |
| paged engine cross b=8 ctx=1024 d=128 fp32 | unresolved | 1.007 | 0.998 | 0.003 |
| paged store store b=32 ctx=1 bf16 | no valid baseline | — | — | — |
| paged store store b=32 ctx=1 fp32 | loss | 1.048 | 0.970 | 0.011 |
| paged store store b=1 ctx=1024 bf16 | win | 0.807 | 1.006 | 0.004 |
| paged store store b=1 ctx=1024 fp32 | unresolved | 1.029 | 1.004 | 0.000 |
| paged engine_out self b=1 ctx=256 bf16 | loss | 1.458 | 0.622 | 0.025 |
| paged engine_out self b=1 ctx=256 fp32 | loss | 1.234 | 1.094 | 0.131 |
| paged engine_out cross b=1 ctx=256 bf16 | loss | 1.534 | 0.617 | 0.020 |
| paged engine_out cross b=1 ctx=256 fp32 | unresolved | 1.028 | 1.106 | 0.069 |
| paged engine_out self b=8 ctx=256 bf16 | unresolved | 1.011 | 0.587 | 0.008 |
| paged engine_out self b=8 ctx=256 fp32 | unresolved | 0.960 | 1.001 | 0.041 |
| paged engine_out cross b=8 ctx=256 bf16 | win | 0.970 | 0.574 | 0.016 |
| paged engine_out cross b=8 ctx=256 fp32 | unresolved | 0.986 | 0.996 | 0.077 |
| paged engine_out self b=8 ctx=1024 bf16 | loss | 1.280 | 0.810 | 0.013 |
| paged engine_out self b=8 ctx=1024 fp32 | win | 0.965 | 1.016 | 0.003 |
| paged engine_out cross b=8 ctx=1024 bf16 | loss | 1.288 | 0.771 | 0.025 |
| paged engine_out cross b=8 ctx=1024 fp32 | unresolved | 0.973 | 1.018 | 0.007 |
| paged engine_out self b=32 ctx=1024 bf16 | loss | 1.126 | 0.843 | 0.026 |
| paged engine_out self b=32 ctx=1024 fp32 | unresolved | 1.018 | 0.967 | 0.006 |
| paged engine_out cross b=32 ctx=1024 bf16 | loss | 1.166 | 0.833 | 0.026 |
| paged engine_out cross b=32 ctx=1024 fp32 | loss | 1.033 | 0.955 | 0.022 |
| paged engine_out self b=1 ctx=4096 bf16 | loss | 1.049 | 0.748 | 0.013 |
| paged engine_out self b=1 ctx=4096 fp32 | unresolved | 1.013 | 1.025 | 0.011 |
| paged engine_out cross b=1 ctx=4096 bf16 | loss | 1.054 | 0.725 | 0.002 |
| paged engine_out cross b=1 ctx=4096 fp32 | unresolved | 0.983 | 1.014 | 0.007 |
| paged api self b=1 ctx=256 bf16 | loss | 1.103 | 0.929 | 0.014 |
| paged api self b=1 ctx=256 fp32 | unresolved | 1.002 | 1.054 | 0.041 |
| paged api cross b=1 ctx=256 bf16 | loss | 1.213 | 0.786 | 0.008 |
| paged api cross b=1 ctx=256 fp32 | unresolved | 1.048 | 1.108 | 0.053 |
| paged api self b=8 ctx=1024 bf16 | loss | 1.179 | 0.794 | 0.010 |
| paged api self b=8 ctx=1024 fp32 | unresolved | 0.993 | 1.020 | 0.007 |
| paged api cross b=8 ctx=1024 bf16 | loss | 1.224 | 0.734 | 0.016 |
| paged api cross b=8 ctx=1024 fp32 | win | 0.956 | 0.985 | 0.024 |

## `paged-full-hilo-scalar-peer.json`

Rows: 92; {'loss': 53, 'unresolved': 27, 'win': 12}.
Invalid new outputs: [].

| Row | Verdict | New / valid C++ | New / previous | Control spread |
|---|---|---:|---:|---:|
| paged engine self b=1 ctx=256 bf16 | loss | 1.565 | 0.755 | 0.032 |
| paged engine self b=1 ctx=256 fp32 | loss | 1.210 | 0.918 | 0.102 |
| paged engine cross b=1 ctx=256 bf16 | loss | 1.596 | 0.708 | 0.039 |
| paged engine cross b=1 ctx=256 fp32 | loss | 1.189 | 0.919 | 0.097 |
| paged engine self b=8 ctx=256 bf16 | loss | 1.068 | 0.595 | 0.021 |
| paged engine self b=8 ctx=256 fp32 | unresolved | 0.999 | 1.023 | 0.035 |
| paged engine cross b=8 ctx=256 bf16 | unresolved | 1.010 | 0.632 | 0.004 |
| paged engine cross b=8 ctx=256 fp32 | loss | 1.060 | 0.998 | 0.029 |
| paged engine self b=8 ctx=1024 bf16 | loss | 1.229 | 0.799 | 0.036 |
| paged engine self b=8 ctx=1024 fp32 | unresolved | 0.978 | 1.076 | 0.048 |
| paged engine cross b=8 ctx=1024 bf16 | loss | 1.218 | 0.806 | 0.018 |
| paged engine cross b=8 ctx=1024 fp32 | unresolved | 0.991 | 1.090 | 0.053 |
| paged engine self b=32 ctx=1024 bf16 | loss | 1.085 | 0.830 | 0.007 |
| paged engine self b=32 ctx=1024 fp32 | unresolved | 1.011 | 1.026 | 0.042 |
| paged engine cross b=32 ctx=1024 bf16 | loss | 1.141 | 0.848 | 0.006 |
| paged engine cross b=32 ctx=1024 fp32 | unresolved | 1.011 | 1.017 | 0.004 |
| paged engine self b=1 ctx=4096 bf16 | loss | 1.115 | 0.758 | 0.030 |
| paged engine self b=1 ctx=4096 fp32 | unresolved | 0.975 | 1.038 | 0.004 |
| paged engine cross b=1 ctx=4096 bf16 | loss | 1.114 | 0.763 | 0.019 |
| paged engine cross b=1 ctx=4096 fp32 | unresolved | 0.970 | 1.016 | 0.021 |
| paged engine self b=8 ctx=255 bf16 | loss | 1.137 | 0.626 | 0.001 |
| paged engine self b=8 ctx=255 fp32 | loss | 1.053 | 1.081 | 0.049 |
| paged engine cross b=8 ctx=255 bf16 | unresolved | 1.026 | 0.638 | 0.027 |
| paged engine cross b=8 ctx=255 fp32 | loss | 1.082 | 1.067 | 0.029 |
| paged engine self b=8 ctx=257 bf16 | loss | 1.056 | 0.639 | 0.018 |
| paged engine self b=8 ctx=257 fp32 | loss | 1.044 | 1.050 | 0.013 |
| paged engine cross b=8 ctx=257 bf16 | loss | 1.060 | 0.621 | 0.006 |
| paged engine cross b=8 ctx=257 fp32 | unresolved | 1.053 | 0.914 | 0.130 |
| paged engine self b=8 ctx=1023 bf16 | loss | 1.216 | 0.832 | 0.013 |
| paged engine self b=8 ctx=1023 fp32 | win | 0.968 | 1.009 | 0.013 |
| paged engine cross b=8 ctx=1023 bf16 | loss | 1.183 | 0.768 | 0.005 |
| paged engine cross b=8 ctx=1023 fp32 | win | 0.966 | 0.997 | 0.016 |
| paged engine self b=8 ctx=1025 bf16 | loss | 1.249 | 0.806 | 0.018 |
| paged engine self b=8 ctx=1025 fp32 | win | 0.942 | 1.002 | 0.001 |
| paged engine cross b=8 ctx=1025 bf16 | loss | 1.245 | 0.771 | 0.005 |
| paged engine cross b=8 ctx=1025 fp32 | unresolved | 0.970 | 1.011 | 0.001 |
| paged engine self b=1 ctx=4097 bf16 | loss | 1.093 | 0.768 | 0.047 |
| paged engine self b=1 ctx=4097 fp32 | unresolved | 0.991 | 1.024 | 0.014 |
| paged engine cross b=1 ctx=4097 bf16 | loss | 1.044 | 0.759 | 0.041 |
| paged engine cross b=1 ctx=4097 fp32 | unresolved | 0.985 | 1.017 | 0.013 |
| paged engine self b=8 ctx=1024 page=16 bf16 | loss | 1.204 | 0.794 | 0.002 |
| paged engine self b=8 ctx=1024 page=16 fp32 | unresolved | 0.971 | 1.005 | 0.023 |
| paged engine self b=8 ctx=1024 page=48 bf16 | loss | 1.240 | 0.675 | 0.001 |
| paged engine self b=8 ctx=1024 page=48 fp32 | win | 0.925 | 0.977 | 0.005 |
| paged engine self b=8 ctx=1024 page=64 bf16 | loss | 1.140 | 0.703 | 0.006 |
| paged engine self b=8 ctx=1024 page=64 fp32 | loss | 1.069 | 1.062 | 0.032 |
| paged engine self b=8 ctx=1024 h=16/4 bf16 | loss | 1.314 | 0.857 | 0.002 |
| paged engine self b=8 ctx=1024 h=16/4 fp32 | win | 0.823 | 1.038 | 0.058 |
| paged engine cross b=8 ctx=1024 h=16/4 bf16 | loss | 1.330 | 0.868 | 0.017 |
| paged engine cross b=8 ctx=1024 h=16/4 fp32 | win | 0.788 | 1.042 | 0.023 |
| paged engine cross b=8 ctx=1024 h=16/1 bf16 | loss | 1.634 | 0.817 | 0.005 |
| paged engine cross b=8 ctx=1024 h=16/1 fp32 | win | 0.960 | 1.026 | 0.009 |
| paged engine self b=8 ctx=het bf16 | win | 0.185 | 0.743 | 0.003 |
| paged engine self b=8 ctx=het fp32 | unresolved | 0.985 | 1.013 | 0.012 |
| paged engine cross b=8 ctx=het bf16 | win | 0.199 | 0.794 | 0.017 |
| paged engine cross b=8 ctx=het fp32 | unresolved | 0.998 | 1.011 | 0.002 |
| paged engine cross b=8 ctx=1024 d=32 bf16 | loss | 1.362 | 0.812 | 0.037 |
| paged engine cross b=8 ctx=1024 d=32 fp32 | unresolved | 0.992 | 1.008 | 0.003 |
| paged engine cross b=8 ctx=1024 d=128 bf16 | loss | 1.456 | 0.776 | 0.064 |
| paged engine cross b=8 ctx=1024 d=128 fp32 | unresolved | 1.017 | 1.000 | 0.006 |
| paged store store b=32 ctx=1 bf16 | loss | 1.056 | 1.031 | 0.005 |
| paged store store b=32 ctx=1 fp32 | loss | 1.159 | 1.030 | 0.056 |
| paged store store b=1 ctx=1024 bf16 | win | 0.762 | 0.996 | 0.007 |
| paged store store b=1 ctx=1024 fp32 | unresolved | 1.026 | 0.997 | 0.001 |
| paged engine_out self b=1 ctx=256 bf16 | loss | 1.442 | 0.661 | 0.038 |
| paged engine_out self b=1 ctx=256 fp32 | loss | 1.113 | 1.044 | 0.026 |
| paged engine_out cross b=1 ctx=256 bf16 | loss | 1.410 | 0.643 | 0.004 |
| paged engine_out cross b=1 ctx=256 fp32 | loss | 1.084 | 0.998 | 0.018 |
| paged engine_out self b=8 ctx=256 bf16 | loss | 1.066 | 0.596 | 0.020 |
| paged engine_out self b=8 ctx=256 fp32 | loss | 1.191 | 1.096 | 0.092 |
| paged engine_out cross b=8 ctx=256 bf16 | unresolved | 0.986 | 0.590 | 0.014 |
| paged engine_out cross b=8 ctx=256 fp32 | unresolved | 1.000 | 0.860 | 0.099 |
| paged engine_out self b=8 ctx=1024 bf16 | loss | 1.231 | 0.805 | 0.020 |
| paged engine_out self b=8 ctx=1024 fp32 | unresolved | 0.975 | 1.019 | 0.037 |
| paged engine_out cross b=8 ctx=1024 bf16 | loss | 1.170 | 0.765 | 0.003 |
| paged engine_out cross b=8 ctx=1024 fp32 | unresolved | 0.978 | 1.008 | 0.029 |
| paged engine_out self b=32 ctx=1024 bf16 | loss | 1.108 | 0.851 | 0.001 |
| paged engine_out self b=32 ctx=1024 fp32 | unresolved | 1.016 | 1.003 | 0.021 |
| paged engine_out cross b=32 ctx=1024 bf16 | loss | 1.115 | 0.874 | 0.047 |
| paged engine_out cross b=32 ctx=1024 fp32 | unresolved | 1.028 | 0.987 | 0.000 |
| paged engine_out self b=1 ctx=4096 bf16 | loss | 1.104 | 0.735 | 0.024 |
| paged engine_out self b=1 ctx=4096 fp32 | win | 0.961 | 1.006 | 0.014 |
| paged engine_out cross b=1 ctx=4096 bf16 | loss | 1.152 | 0.752 | 0.001 |
| paged engine_out cross b=1 ctx=4096 fp32 | win | 0.932 | 1.015 | 0.018 |
| paged api self b=1 ctx=256 bf16 | loss | 1.140 | 0.924 | 0.002 |
| paged api self b=1 ctx=256 fp32 | loss | 1.094 | 1.061 | 0.048 |
| paged api cross b=1 ctx=256 bf16 | loss | 1.199 | 0.789 | 0.012 |
| paged api cross b=1 ctx=256 fp32 | loss | 1.175 | 1.041 | 0.048 |
| paged api self b=8 ctx=1024 bf16 | loss | 1.103 | 0.778 | 0.004 |
| paged api self b=8 ctx=1024 fp32 | unresolved | 0.970 | 1.010 | 0.001 |
| paged api cross b=8 ctx=1024 bf16 | loss | 1.194 | 0.785 | 0.013 |
| paged api cross b=8 ctx=1024 fp32 | unresolved | 0.971 | 1.026 | 0.013 |

## `paged-full-hilo-rotated.json`

Rows: 92; {'loss': 51, 'unresolved': 20, 'win': 21}.
Invalid new outputs: [].

| Row | Verdict | New / valid C++ | New / previous | Control spread |
|---|---|---:|---:|---:|
| paged engine self b=1 ctx=256 bf16 | loss | 1.700 | 0.714 | 0.005 |
| paged engine self b=1 ctx=256 fp32 | loss | 1.487 | 0.769 | 0.021 |
| paged engine cross b=1 ctx=256 bf16 | loss | 1.788 | 0.696 | 0.007 |
| paged engine cross b=1 ctx=256 fp32 | loss | 1.545 | 0.779 | 0.001 |
| paged engine self b=8 ctx=256 bf16 | loss | 1.150 | 0.566 | 0.013 |
| paged engine self b=8 ctx=256 fp32 | unresolved | 1.062 | 1.013 | 0.103 |
| paged engine cross b=8 ctx=256 bf16 | loss | 1.182 | 0.559 | 0.001 |
| paged engine cross b=8 ctx=256 fp32 | loss | 1.063 | 1.010 | 0.052 |
| paged engine self b=8 ctx=1024 bf16 | loss | 1.104 | 0.731 | 0.015 |
| paged engine self b=8 ctx=1024 fp32 | win | 0.937 | 1.030 | 0.000 |
| paged engine cross b=8 ctx=1024 bf16 | loss | 1.205 | 0.693 | 0.031 |
| paged engine cross b=8 ctx=1024 fp32 | win | 0.919 | 1.008 | 0.011 |
| paged engine self b=32 ctx=1024 bf16 | loss | 1.195 | 0.856 | 0.042 |
| paged engine self b=32 ctx=1024 fp32 | win | 0.964 | 0.981 | 0.001 |
| paged engine cross b=32 ctx=1024 bf16 | loss | 1.151 | 0.857 | 0.001 |
| paged engine cross b=32 ctx=1024 fp32 | unresolved | 1.003 | 1.015 | 0.011 |
| paged engine self b=1 ctx=4096 bf16 | loss | 1.068 | 0.731 | 0.025 |
| paged engine self b=1 ctx=4096 fp32 | win | 0.956 | 1.005 | 0.018 |
| paged engine cross b=1 ctx=4096 bf16 | loss | 1.086 | 0.763 | 0.010 |
| paged engine cross b=1 ctx=4096 fp32 | win | 0.955 | 1.014 | 0.029 |
| paged engine self b=8 ctx=255 bf16 | loss | 1.228 | 0.599 | 0.002 |
| paged engine self b=8 ctx=255 fp32 | unresolved | 1.008 | 0.980 | 0.064 |
| paged engine cross b=8 ctx=255 bf16 | loss | 1.199 | 0.574 | 0.000 |
| paged engine cross b=8 ctx=255 fp32 | unresolved | 0.986 | 0.945 | 0.011 |
| paged engine self b=8 ctx=257 bf16 | loss | 1.207 | 0.596 | 0.003 |
| paged engine self b=8 ctx=257 fp32 | unresolved | 1.025 | 1.033 | 0.037 |
| paged engine cross b=8 ctx=257 bf16 | loss | 1.158 | 0.603 | 0.015 |
| paged engine cross b=8 ctx=257 fp32 | loss | 1.063 | 1.058 | 0.012 |
| paged engine self b=8 ctx=1023 bf16 | loss | 1.075 | 0.754 | 0.039 |
| paged engine self b=8 ctx=1023 fp32 | unresolved | 0.927 | 1.037 | 0.106 |
| paged engine cross b=8 ctx=1023 bf16 | loss | 1.067 | 0.695 | 0.027 |
| paged engine cross b=8 ctx=1023 fp32 | win | 0.944 | 1.049 | 0.052 |
| paged engine self b=8 ctx=1025 bf16 | loss | 1.137 | 0.739 | 0.002 |
| paged engine self b=8 ctx=1025 fp32 | unresolved | 0.926 | 1.034 | 0.103 |
| paged engine cross b=8 ctx=1025 bf16 | loss | 1.046 | 0.700 | 0.021 |
| paged engine cross b=8 ctx=1025 fp32 | unresolved | 0.949 | 1.016 | 0.093 |
| paged engine self b=1 ctx=4097 bf16 | loss | 1.091 | 0.772 | 0.025 |
| paged engine self b=1 ctx=4097 fp32 | unresolved | 0.999 | 0.982 | 0.002 |
| paged engine cross b=1 ctx=4097 bf16 | loss | 1.121 | 0.823 | 0.011 |
| paged engine cross b=1 ctx=4097 fp32 | unresolved | 0.971 | 1.026 | 0.017 |
| paged engine self b=8 ctx=1024 page=16 bf16 | loss | 1.160 | 0.778 | 0.007 |
| paged engine self b=8 ctx=1024 page=16 fp32 | unresolved | 0.942 | 0.967 | 0.062 |
| paged engine self b=8 ctx=1024 page=48 bf16 | loss | 1.186 | 0.627 | 0.023 |
| paged engine self b=8 ctx=1024 page=48 fp32 | win | 0.933 | 0.976 | 0.035 |
| paged engine self b=8 ctx=1024 page=64 bf16 | loss | 1.292 | 0.742 | 0.002 |
| paged engine self b=8 ctx=1024 page=64 fp32 | unresolved | 1.033 | 1.006 | 0.051 |
| paged engine self b=8 ctx=1024 h=16/4 bf16 | loss | 1.261 | 0.814 | 0.004 |
| paged engine self b=8 ctx=1024 h=16/4 fp32 | win | 0.740 | 1.019 | 0.035 |
| paged engine cross b=8 ctx=1024 h=16/4 bf16 | loss | 1.333 | 0.830 | 0.024 |
| paged engine cross b=8 ctx=1024 h=16/4 fp32 | win | 0.760 | 1.063 | 0.016 |
| paged engine cross b=8 ctx=1024 h=16/1 bf16 | loss | 1.655 | 0.813 | 0.001 |
| paged engine cross b=8 ctx=1024 h=16/1 fp32 | win | 0.961 | 0.993 | 0.001 |
| paged engine self b=8 ctx=het bf16 | win | 0.217 | 0.820 | 0.029 |
| paged engine self b=8 ctx=het fp32 | win | 0.916 | 0.991 | 0.003 |
| paged engine cross b=8 ctx=het bf16 | win | 0.197 | 0.782 | 0.010 |
| paged engine cross b=8 ctx=het fp32 | win | 0.914 | 0.986 | 0.010 |
| paged engine cross b=8 ctx=1024 d=32 bf16 | loss | 1.152 | 0.686 | 0.038 |
| paged engine cross b=8 ctx=1024 d=32 fp32 | unresolved | 0.985 | 1.023 | 0.027 |
| paged engine cross b=8 ctx=1024 d=128 bf16 | loss | 1.282 | 0.739 | 0.066 |
| paged engine cross b=8 ctx=1024 d=128 fp32 | win | 0.918 | 0.974 | 0.041 |
| paged store store b=32 ctx=1 bf16 | unresolved | 0.983 | 1.010 | 0.065 |
| paged store store b=32 ctx=1 fp32 | loss | 1.043 | 0.962 | 0.010 |
| paged store store b=1 ctx=1024 bf16 | win | 0.762 | 1.000 | 0.017 |
| paged store store b=1 ctx=1024 fp32 | loss | 1.041 | 1.006 | 0.007 |
| paged engine_out self b=1 ctx=256 bf16 | loss | 1.585 | 0.653 | 0.001 |
| paged engine_out self b=1 ctx=256 fp32 | loss | 1.259 | 1.032 | 0.018 |
| paged engine_out cross b=1 ctx=256 bf16 | loss | 1.533 | 0.593 | 0.006 |
| paged engine_out cross b=1 ctx=256 fp32 | loss | 1.104 | 0.851 | 0.051 |
| paged engine_out self b=8 ctx=256 bf16 | loss | 1.225 | 0.546 | 0.011 |
| paged engine_out self b=8 ctx=256 fp32 | unresolved | 1.038 | 0.965 | 0.040 |
| paged engine_out cross b=8 ctx=256 bf16 | loss | 1.206 | 0.553 | 0.019 |
| paged engine_out cross b=8 ctx=256 fp32 | win | 0.957 | 0.965 | 0.021 |
| paged engine_out self b=8 ctx=1024 bf16 | loss | 1.112 | 0.763 | 0.002 |
| paged engine_out self b=8 ctx=1024 fp32 | win | 0.916 | 0.998 | 0.058 |
| paged engine_out cross b=8 ctx=1024 bf16 | loss | 1.123 | 0.713 | 0.017 |
| paged engine_out cross b=8 ctx=1024 fp32 | unresolved | 0.932 | 1.015 | 0.084 |
| paged engine_out self b=32 ctx=1024 bf16 | loss | 1.154 | 0.831 | 0.032 |
| paged engine_out self b=32 ctx=1024 fp32 | win | 0.948 | 0.965 | 0.014 |
| paged engine_out cross b=32 ctx=1024 bf16 | loss | 1.111 | 0.862 | 0.021 |
| paged engine_out cross b=32 ctx=1024 fp32 | win | 0.959 | 0.995 | 0.027 |
| paged engine_out self b=1 ctx=4096 bf16 | loss | 1.064 | 0.809 | 0.015 |
| paged engine_out self b=1 ctx=4096 fp32 | win | 0.950 | 1.009 | 0.025 |
| paged engine_out cross b=1 ctx=4096 bf16 | loss | 1.113 | 0.787 | 0.001 |
| paged engine_out cross b=1 ctx=4096 fp32 | unresolved | 1.001 | 1.013 | 0.001 |
| paged api self b=1 ctx=256 bf16 | loss | 1.056 | 0.892 | 0.019 |
| paged api self b=1 ctx=256 fp32 | loss | 1.058 | 1.025 | 0.002 |
| paged api cross b=1 ctx=256 bf16 | loss | 1.212 | 0.788 | 0.003 |
| paged api cross b=1 ctx=256 fp32 | unresolved | 1.039 | 0.934 | 0.071 |
| paged api self b=8 ctx=1024 bf16 | loss | 1.190 | 0.806 | 0.022 |
| paged api self b=8 ctx=1024 fp32 | unresolved | 0.986 | 1.016 | 0.022 |
| paged api cross b=8 ctx=1024 bf16 | loss | 1.097 | 0.705 | 0.015 |
| paged api cross b=8 ctx=1024 fp32 | unresolved | 0.985 | 1.016 | 0.031 |

## `flash-full-rq128.json`

Rows: 78; {'loss': 2, 'unresolved': 22, 'win': 54}.
Invalid new outputs: [].

| Row | Verdict | New / valid C++ | New / previous | Control spread |
|---|---|---:|---:|---:|
| flash 128:8:64 bf16 | loss | 1.079 | 1.103 | 0.044 |
| flash 128:8:64 fp32 | unresolved | 1.014 | 1.036 | 0.001 |
| flash 512:8:64 bf16 | unresolved | 0.982 | 1.039 | 0.004 |
| flash 512:8:64 fp32 | win | 0.967 | 1.013 | 0.006 |
| flash 1024:8:64 bf16 | win | 0.917 | 1.010 | 0.002 |
| flash 1024:8:64 fp32 | win | 0.961 | 1.007 | 0.003 |
| flash 2048:8:64 bf16 | win | 0.921 | 1.001 | 0.001 |
| flash 2048:8:64 fp32 | win | 0.953 | 1.005 | 0.000 |
| flash 4096:8:64 bf16 | win | 0.937 | 1.005 | 0.002 |
| flash 4096:8:64 fp32 | win | 0.957 | 1.001 | 0.002 |
| flash 256:16:128 bf16 | unresolved | 0.995 | 1.031 | 0.001 |
| flash 256:16:128 fp32 | unresolved | 0.996 | 1.013 | 0.000 |
| flash 1024:16:128 bf16 | win | 0.927 | 1.004 | 0.001 |
| flash 1024:16:128 fp32 | win | 0.913 | 1.001 | 0.002 |
| flash 2048:16:128 bf16 | win | 0.930 | 1.006 | 0.000 |
| flash 2048:16:128 fp32 | win | 0.961 | 0.936 | 0.002 |
| flash 4096:16:128 bf16 | win | 0.932 | 0.974 | 0.002 |
| flash 4096:16:128 fp32 | unresolved | 0.976 | 0.906 | 0.001 |
| flash 512:32:128 bf16 | unresolved | 0.979 | 1.003 | 0.005 |
| flash 512:32:128 fp32 | unresolved | 0.981 | 1.008 | 0.002 |
| flash 1024:8:32 bf16 | win | 0.944 | 1.015 | 0.004 |
| flash 1024:8:32 fp32 | win | 0.954 | 1.016 | 0.003 |
| flash 17:8:64 bf16 | win | 0.922 | 1.053 | 0.038 |
| flash 17:8:64 fp32 | win | 0.825 | 1.019 | 0.033 |
| flash 65:8:64 bf16 | unresolved | 0.982 | 1.083 | 0.012 |
| flash 65:8:64 fp32 | win | 0.914 | 1.087 | 0.002 |
| flash 129:8:64 bf16 | unresolved | 1.027 | 1.072 | 0.027 |
| flash 129:8:64 fp32 | win | 0.957 | 1.056 | 0.004 |
| flash 257:8:64 bf16 | unresolved | 1.019 | 1.053 | 0.015 |
| flash 257:8:64 fp32 | win | 0.954 | 1.031 | 0.004 |
| flash 513:8:64 bf16 | win | 0.961 | 1.035 | 0.002 |
| flash 513:8:64 fp32 | unresolved | 0.982 | 1.028 | 0.003 |
| flash 1025:8:64 bf16 | win | 0.935 | 1.013 | 0.001 |
| flash 1025:8:64 fp32 | win | 0.953 | 1.004 | 0.001 |
| flash 2049:8:64 bf16 | win | 0.922 | 1.003 | 0.001 |
| flash 2049:8:64 fp32 | win | 0.969 | 1.007 | 0.002 |
| flash 4097:8:64 bf16 | win | 0.918 | 1.001 | 0.000 |
| flash 4097:8:64 fp32 | win | 0.953 | 1.000 | 0.000 |
| flash 129:16:128 bf16 | unresolved | 1.006 | 1.095 | 0.022 |
| flash 129:16:128 fp32 | win | 0.960 | 1.068 | 0.008 |
| flash 257:16:128 bf16 | unresolved | 1.013 | 1.050 | 0.009 |
| flash 257:16:128 fp32 | unresolved | 0.979 | 1.030 | 0.003 |
| flash 1025:16:128 bf16 | win | 0.926 | 1.010 | 0.001 |
| flash 1025:16:128 fp32 | unresolved | 0.977 | 1.010 | 0.006 |
| flash 2049:16:128 bf16 | win | 0.920 | 1.001 | 0.002 |
| flash 2049:16:128 fp32 | unresolved | 0.970 | 0.966 | 0.002 |
| flash 1025:8:32 bf16 | win | 0.951 | 1.011 | 0.002 |
| flash 1025:8:32 fp32 | win | 0.946 | 1.013 | 0.002 |
| flash 1x129:8:64 bf16 | win | 0.897 | 1.184 | 0.096 |
| flash 1x129:8:64 fp32 | win | 0.725 | 1.070 | 0.015 |
| flash 17x301:8:64 bf16 | unresolved | 0.979 | 1.052 | 0.025 |
| flash 17x301:8:64 fp32 | win | 0.860 | 1.029 | 0.001 |
| flash 127x129:8:64 bf16 | unresolved | 1.003 | 1.022 | 0.010 |
| flash 127x129:8:64 fp32 | win | 0.943 | 1.020 | 0.015 |
| flash 129x127:8:64 bf16 | loss | 1.054 | 1.086 | 0.042 |
| flash 129x127:8:64 fp32 | win | 0.953 | 1.012 | 0.014 |
| flash 257x513:8:64 bf16 | win | 0.958 | 1.000 | 0.005 |
| flash 257x513:8:64 fp32 | win | 0.950 | 1.017 | 0.001 |
| flash 513x257:8:64 bf16 | win | 0.969 | 1.021 | 0.013 |
| flash 513x257:8:64 fp32 | win | 0.962 | 1.032 | 0.003 |
| flash 1025x2049:8:64 bf16 | win | 0.916 | 1.002 | 0.002 |
| flash 1025x2049:8:64 fp32 | win | 0.944 | 1.004 | 0.001 |
| flash 2049x1025:8:64 bf16 | win | 0.941 | 1.005 | 0.002 |
| flash 2049x1025:8:64 fp32 | win | 0.963 | 1.012 | 0.006 |
| flash 1024:8:64 noncausal bf16 | win | 0.934 | 1.011 | 0.002 |
| flash 1024:8:64 noncausal fp32 | win | 0.962 | 1.006 | 0.003 |
| flash 1025:8:64 noncausal bf16 | win | 0.932 | 1.009 | 0.002 |
| flash 1025:8:64 noncausal fp32 | win | 0.957 | 1.003 | 0.001 |
| flash 1024:16:128 noncausal bf16 | win | 0.933 | 1.006 | 0.002 |
| flash 1024:16:128 noncausal fp32 | win | 0.878 | 1.004 | 0.001 |
| flash 1024:32/8:128 bf16 | win | 0.943 | 1.009 | 0.001 |
| flash 1024:32/8:128 fp32 | unresolved | 0.984 | 1.003 | 0.000 |
| flash 1024:16/1:64 bf16 | win | 0.935 | 1.005 | 0.003 |
| flash 1024:16/1:64 fp32 | win | 0.954 | 1.003 | 0.001 |
| flash 1024:8:64 fused bf16 | win | 0.917 | 1.012 | 0.000 |
| flash 1024:8:64 fused fp32 | unresolved | 0.972 | 1.002 | 0.002 |
| flash 257:16:128 fused bf16 | unresolved | 1.013 | 1.036 | 0.006 |
| flash 257:16:128 fused fp32 | unresolved | 0.985 | 1.034 | 0.001 |
