import glob
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "benchmarks"), os.path.join(ROOT, "tests")]
import bench
import arms as A

variant = os.path.join(ROOT, "reports/axion/arm-validation/rq128-package")
os.makedirs(variant, exist_ok=True)
for path in glob.glob(A.PKG + "/*.py") + glob.glob(A.PKG + "/_C*.so"):
    shutil.copy(path, variant)
path = os.path.join(variant, "_kernel.py")
with open(path) as f:
    source = f.read()
old = "rq = max(16, min(128, 16384 // (hd * 4)))"
assert source.count(old) == 1
with open(path, "w") as f:
    f.write(source.replace(old, "rq = 128 if hd == 128 else max(16, min(128, 16384 // (hd * 4)))"))
A.PKG = variant
raise SystemExit(bench.main())
