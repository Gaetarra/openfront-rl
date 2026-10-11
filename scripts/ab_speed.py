"""A code A/B of PPO's speed (PLAN 4A, "Speed work"): the speed check's probe of the 3B clone on the old code (a
worktree of the commit before the change) and on the new code (this checkout), in the order old, new, new, old, back
to back, after a warm-up probe of the old code that is not counted (lever 1's first A/B started from a PC cold after
hours idle, and its first probe ran fast). Then the rule fixed before the runs: the change is credited with the mean
gain if both new probes beat both old ones and the gain is >= 3%; each probe's own GPU peak must stay <= 9,400 MiB.
PC idle, ~18 min:

  .venv/Scripts/python scripts/ab_speed.py --old pins/ab-base --label lever1

The old worktree: git worktree add --detach pins/ab-base <commit>, then junctions to this checkout's game folder,
checkpoints and runs (CLAUDE.md, "Benchmarks"). Logs and the verdict: runs/ab/.
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = ["scripts/speed_check.py", "--ppo-minibatches", "12", "--skip-bc", "--skip-v1", "--v4",
         "checkpoints/bc3b-v4/latest.pt"]
MIN_GAIN, MAX_PEAK = 0.03, 9400


def verdict(old, new, peaks):
    """The rule, on the probes' decisions/s and own GPU peaks (MiB)."""
    gain = sum(new) / sum(old) - 1
    credited = min(new) > max(old) and gain >= MIN_GAIN and max(peaks) <= MAX_PEAK
    return {"gain": gain, "credited": credited, "new_beats_old": min(new) > max(old), "max_peak_mib": max(peaks)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--old", required=True, help="the worktree of the old code, e.g. pins/ab-base")
    p.add_argument("--label", required=True, help="the change's name, for the logs")
    a = p.parse_args()
    old = (ROOT / a.old).resolve()
    out = ROOT / "runs" / "ab"
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    rates, peaks = {"warm-up": [], "old": [], "new": []}, []
    for i, arm in enumerate(("warm-up", "old", "new", "new", "old")):
        log = out / f"{stamp}-{a.label}-{i}-{arm}.log"
        with open(log, "w") as f:
            subprocess.run([sys.executable, *PROBE], cwd=ROOT if arm == "new" else old, stdout=f,
                           stderr=subprocess.STDOUT, check=True)
        text = log.read_text()
        rate = float(re.search(r"PPO, v4: ([\d.]+) decisions/s", text).group(1))
        peak = int(re.search(r"its own peak (-?\d+) MiB", text).group(1))
        rates[arm].append(rate)
        note = " (not counted)" if i == 0 else ""
        print(f"{i} {arm}: {rate:.1f} decisions/s, own peak {peak} MiB{note}", flush=True)
        if i > 0:
            peaks.append(peak)
    result = {"label": a.label, "old_code": str(old), "warm_up": rates["warm-up"], "old": rates["old"],
              "new": rates["new"], "peaks_mib": peaks, **verdict(rates["old"], rates["new"], peaks)}
    (out / f"{stamp}-{a.label}.json").write_text(json.dumps(result, indent=1))
    print(f"{a.label}: {'CREDITED' if result['credited'] else 'no measurable gain'}, mean gain {result['gain']:+.1%} "
          f"(both new above both old: {result['new_beats_old']}; max own peak {result['max_peak_mib']} MiB)")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--self-check"]:
        assert verdict([100, 102], [106, 107], [9000] * 4)["credited"]
        assert not verdict([100, 106], [105, 107], [9000] * 4)["credited"]  # a new probe below an old one
        assert not verdict([100, 101], [102, 103], [9000] * 4)["credited"]  # under 3%
        assert not verdict([100, 101], [110, 111], [9000, 9500, 9000, 9000])["credited"]  # memory
        print("ok")
    else:
        main()
