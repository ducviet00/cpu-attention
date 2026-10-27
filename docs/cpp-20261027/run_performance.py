"""Restore every CPU governor even when a benchmark fails."""
import glob
import subprocess
import sys

saved = {p: open(p).read().strip() for p in glob.glob(
    "/sys/devices/system/cpu/cpufreq/policy*/scaling_governor")}


def set_governor(path, value):
    subprocess.run(["sudo", "-n", "tee", path], input=value + "\n",
                   text=True, stdout=subprocess.DEVNULL, check=True)


try:
    for path in saved:
        set_governor(path, "performance")
    result = subprocess.run(sys.argv[1:])
finally:
    for path, value in saved.items():
        set_governor(path, value)
sys.exit(result.returncode)
