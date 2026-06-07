"""
Tiny thermal-throttle helper for long-running scripts on the Pi.

Reads `/sys/class/thermal/thermal_zone0/temp` (no sudo, no vcgencmd
dependency). Pi 5 hard-throttles at 80 °C; we yield earlier and let it
cool to a safe floor before resuming.

Usage:
    import thermal
    thermal.cool_if_hot()                      # default thresholds
    print(thermal.cpu_c())                     # one-off read

Drop `thermal.cool_if_hot()` between heavy phases (per fold, per model)
in the run scripts. No-ops on non-Pi hosts (returns 0.0 / sleeps never).
"""

import time
from pathlib import Path

THERMAL_FILE = Path('/sys/class/thermal/thermal_zone0/temp')

# Defaults: yield at 75 °C, resume only when back to ≤ 70 °C.
HOT_C    = 75.0
COOL_C   = 70.0
SLEEP_S  = 30
MAX_WAIT_S = 900       # 15 min hard cap before giving up


def cpu_c() -> float:
    """Returns CPU temp °C, or 0.0 sentinel if the thermal interface is missing
    (e.g. WSL or any non-Pi host) so cool_if_hot() becomes a no-op."""
    if not THERMAL_FILE.exists():
        return 0.0
    try:
        return float(THERMAL_FILE.read_text().strip()) / 1000.0
    except (OSError, ValueError):
        return 0.0


def cool_if_hot(hot_c: float = HOT_C,
                cool_c: float = COOL_C,
                sleep_s: int = SLEEP_S,
                max_wait_s: int = MAX_WAIT_S,
                tag: str = '') -> int:
    """
    If CPU is above `hot_c`, sleep `sleep_s` at a time until it drops to
    `cool_c` or `max_wait_s` elapses. Returns total seconds slept.
    """
    waited = 0
    t = cpu_c()
    if t <= hot_c:
        return 0

    label = f'[thermal{(":" + tag) if tag else ""}]'
    print(f'{label} {t:.1f}°C > {hot_c}°C — cooling down (target ≤{cool_c}°C)…',
          flush=True)
    while t > cool_c and waited < max_wait_s:
        time.sleep(sleep_s)
        waited += sleep_s
        t = cpu_c()
        print(f'{label}   …after {waited}s: {t:.1f}°C', flush=True)
    if waited >= max_wait_s and t > cool_c:
        print(f'{label} WARN: still {t:.1f}°C after {waited}s, continuing anyway',
              flush=True)
    else:
        print(f'{label} resumed at {t:.1f}°C (slept {waited}s)', flush=True)
    return waited


if __name__ == '__main__':
    print(f'CPU: {cpu_c():.1f}°C')
