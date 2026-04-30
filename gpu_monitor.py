#!/usr/bin/env python3
r"""
Live GPU usage monitor — two leaderboards, one for compute % and one for VRAM.

Why two queries?
  Per-process compute %: nvidia-smi pmon. Fast. Fine for graphics + compute.
  Per-process VRAM:      Windows GPU performance counters via PowerShell
                         Get-Counter on "\GPU Process Memory(*)\Local Usage".
                         The NVIDIA driver doesn't expose VRAM under WDDM
                         ("Not available in WDDM driver model" from
                         nvidia-smi -q), so we go through the same Win32
                         counters that Task Manager uses.

The PowerShell counter call takes ~1.7s, so the default refresh is 3s —
fast enough to see contention, slow enough that PowerShell startup
doesn't dominate.

Usage:
    python gpu_monitor.py                  # 3s refresh, top 8 each
    python gpu_monitor.py --interval 5     # 5s refresh
    python gpu_monitor.py --top 5          # 5 entries per list
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional


def nvidia_smi_path() -> Optional[str]:
    p = shutil.which('nvidia-smi')
    if p:
        return p
    candidate = r'C:\Windows\System32\nvidia-smi.exe'
    if os.path.exists(candidate):
        return candidate
    return None


# ─── nvidia-smi queries ──────────────────────────────────────────────────────

def query_gpu_summary(smi: str) -> Dict:
    """util/mem/power/temp via --query-gpu (fast)."""
    cmd = [smi, '--query-gpu=utilization.gpu,memory.used,memory.total,'
                 'power.draw,temperature.gpu',
           '--format=csv,noheader,nounits']
    out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    util, mem_u, mem_t, pwr, temp = (s.strip() for s in out.split(','))
    return {
        'util_pct': int(float(util)),
        'mem_used_mib': int(float(mem_u)),
        'mem_total_mib': int(float(mem_t)),
        'power_w': float(pwr) if pwr.replace('.', '', 1).isdigit() else 0.0,
        'temp_c': int(float(temp)),
    }


def query_processes_pmon(smi: str) -> Dict[int, Dict]:
    """Fast per-process query via `nvidia-smi pmon -c 1 -s u`. Returns
    {pid: {name, type, sm_pct}} for active processes. Memory column from
    pmon is unreliable on WDDM (mostly 0 for graphics procs), so we get
    VRAM separately."""
    procs: Dict[int, Dict] = {}
    try:
        out = subprocess.check_output(
            [smi, 'pmon', '-c', '1', '-s', 'u'],
            text=True, stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return procs

    for line in out.splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        parts = line.split()
        try:
            pid = int(parts[1])
        except (IndexError, ValueError):
            continue
        ptype = parts[2] if len(parts) > 2 else '?'
        def to_int(s: str) -> int:
            try:
                return int(s)
            except ValueError:
                return 0
        sm = to_int(parts[3]) if len(parts) > 3 else 0
        # Last column is the command basename in pmon output
        name = parts[-1] if len(parts) >= 6 else '?'
        # pmon may emit the same PID with different schedulers — keep max sm%
        if pid in procs:
            procs[pid]['sm_pct'] = max(procs[pid]['sm_pct'], sm)
        else:
            procs[pid] = {'name': name, 'type': ptype, 'sm_pct': sm}
    return procs


def query_processes_full(smi: str) -> Dict[int, str]:
    """Map every PID nvidia-smi sees to a short process name. Uses
    `nvidia-smi -q --display=PIDS` (slower than pmon but covers ALL
    GPU-using processes, including idle graphics ones). Used to populate
    names for VRAM-only entries that pmon misses."""
    pid_to_name: Dict[int, str] = {}
    try:
        out = subprocess.check_output(
            [smi, '-q', '--display=PIDS'],
            text=True, stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return pid_to_name

    cur_pid: Optional[int] = None
    for line in out.splitlines():
        line = line.strip()
        if line.startswith('Process ID'):
            try:
                cur_pid = int(line.split(':', 1)[1].strip())
            except (IndexError, ValueError):
                cur_pid = None
        elif line.startswith('Name') and cur_pid is not None:
            name = line.split(':', 1)[1].strip()
            pid_to_name[cur_pid] = os.path.basename(name) if name else '?'
            cur_pid = None
    return pid_to_name


# ─── Windows perf-counter VRAM query ─────────────────────────────────────────

_VRAM_PS_SCRIPT = (
    '$ErrorActionPreference="SilentlyContinue";'
    '(Get-Counter "\\GPU Process Memory(*)\\Local Usage").CounterSamples |'
    ' Where-Object { $_.CookedValue -gt 0 } |'
    ' ForEach-Object {'
    '  if ($_.InstanceName -match "pid_(\\d+)_") {'
    '    "{0},{1}" -f $matches[1], $_.CookedValue'
    '  }'
    ' }'
)


def query_pid_names_via_powershell(pids: List[int],
                                    timeout: float = 5.0) -> Dict[int, str]:
    """Fallback name lookup for PIDs nvidia-smi reports without a name
    (e.g. dwm.exe, system services). Returns {pid: image_name}.
    Empty on non-Windows."""
    if not sys.platform.startswith('win') or not pids:
        return {}
    pid_list = ','.join(str(p) for p in pids)
    script = (
        '$ErrorActionPreference="SilentlyContinue";'
        f'Get-Process -Id {pid_list} |'
        ' ForEach-Object { "{0},{1}" -f $_.Id, $_.ProcessName }'
    )
    try:
        out = subprocess.check_output(
            ['powershell', '-NoProfile', '-Command', script],
            text=True, stderr=subprocess.DEVNULL, timeout=timeout,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {}

    result: Dict[int, str] = {}
    for line in out.splitlines():
        line = line.strip()
        if not line or ',' not in line:
            continue
        try:
            pid_s, name = line.split(',', 1)
            result[int(pid_s)] = name + '.exe'
        except ValueError:
            continue
    return result


def query_vram_per_pid(timeout: float = 10.0) -> Dict[int, int]:
    """Per-process VRAM in MiB via Windows GPU perf counters.

    Returns {pid: mib}. Empty on non-Windows or when the counter isn't
    available (PowerShell is slow — ~1.5-2s — so we only call this once
    per refresh).
    """
    if not sys.platform.startswith('win'):
        return {}
    try:
        out = subprocess.check_output(
            ['powershell', '-NoProfile', '-Command', _VRAM_PS_SCRIPT],
            text=True, stderr=subprocess.DEVNULL, timeout=timeout,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {}

    pid_to_bytes: Dict[int, int] = {}
    for line in out.splitlines():
        line = line.strip()
        if not line or ',' not in line:
            continue
        try:
            pid_s, bytes_s = line.split(',', 1)
            pid = int(pid_s)
            bytes_v = int(float(bytes_s))
        except (ValueError, IndexError):
            continue
        # A single PID can show up under multiple LUIDs (different
        # graphics adapters / phys instances) — sum across all of them
        pid_to_bytes[pid] = pid_to_bytes.get(pid, 0) + bytes_v
    return {pid: b // (1024 * 1024) for pid, b in pid_to_bytes.items()}


# ─── Rendering ───────────────────────────────────────────────────────────────

def render(summary: Dict, processes: Dict[int, Dict], top_n: int) -> str:
    """processes is a {pid: {name, type, sm_pct, vram_mib}} merged map."""
    util_bar = '#' * (summary['util_pct'] // 5)
    util_bar = util_bar.ljust(20)
    mem_pct = 100.0 * summary['mem_used_mib'] / max(1, summary['mem_total_mib'])
    header = (f"GPU util: {summary['util_pct']:>3d}% [{util_bar}] | "
              f"VRAM: {summary['mem_used_mib']:>5d}/{summary['mem_total_mib']:>5d} MB "
              f"({mem_pct:>4.1f}%) | "
              f"{summary['power_w']:>5.1f} W | {summary['temp_c']:>3d} C")

    by_sm = sorted(processes.values(),
                   key=lambda p: (p['sm_pct'], p.get('vram_mib', 0)),
                   reverse=True)
    by_vram = sorted(processes.values(),
                     key=lambda p: (p.get('vram_mib', 0), p['sm_pct']),
                     reverse=True)

    def fmt_sm_row(p: Dict) -> str:
        mark = '>>' if p['sm_pct'] >= 50 else '  '
        return (f"{mark}{p['pid']:>7}  {p['type']:<4}  "
                f"{p['sm_pct']:>3d}%  {p['name'][:28]}")

    def fmt_vram_row(p: Dict) -> str:
        v = p.get('vram_mib', 0)
        mark = '>>' if v >= 1024 else '  '   # >1 GB tagged
        return (f"{mark}{p['pid']:>7}  {p['type']:<4}  "
                f"{v:>6d}MB  {p['name'][:28]}")

    lines = [header, '']
    lines.append('TOP BY GPU COMPUTE')
    lines.append(f"  {'PID':>7}  {'TYPE':<4}  {'SM%':>4}  PROCESS")
    if not by_sm:
        lines.append('  (no processes reported)')
    else:
        active_sm = [p for p in by_sm if p['sm_pct'] > 0]
        for p in (active_sm or by_sm)[:top_n]:
            lines.append(fmt_sm_row(p))
        if not active_sm:
            lines.append('  (all reporting 0% — no compute load right now)')

    lines.append('')
    lines.append('TOP BY VRAM')
    lines.append(f"  {'PID':>7}  {'TYPE':<4}  {'  VRAM':>8}  PROCESS")
    if not by_vram or all(p.get('vram_mib', 0) == 0 for p in by_vram):
        lines.append('  (perf counter unavailable; needs Windows + PowerShell)')
    else:
        for p in by_vram[:top_n]:
            if p.get('vram_mib', 0) == 0:
                break
            lines.append(fmt_vram_row(p))

    return '\n'.join(lines)


def merge_process_data(pmon_procs: Dict[int, Dict],
                       full_names: Dict[int, str],
                       vram: Dict[int, int]) -> Dict[int, Dict]:
    """Combine the three source maps into a single per-PID dict."""
    out: Dict[int, Dict] = {}
    all_pids = set(pmon_procs) | set(full_names) | set(vram)
    for pid in all_pids:
        info = pmon_procs.get(pid, {}).copy()
        info['pid'] = pid
        if 'name' not in info or info.get('name') == '?':
            info['name'] = full_names.get(pid, info.get('name', '?'))
        info.setdefault('type', 'C+G')
        info.setdefault('sm_pct', 0)
        info['vram_mib'] = vram.get(pid, 0)
        out[pid] = info
    return out


def main():
    parser = argparse.ArgumentParser(description='Live GPU process monitor')
    parser.add_argument('--interval', type=float, default=3.0,
                        help='Refresh interval in seconds (default: 3 — '
                             'PowerShell perf-counter call takes ~1.7s)')
    parser.add_argument('--top', type=int, default=8,
                        help='Show top N entries in each list (default: 8)')
    parser.add_argument('--no-vram', action='store_true',
                        help='Skip the slow PowerShell VRAM query')
    args = parser.parse_args()

    smi = nvidia_smi_path()
    if smi is None:
        print('nvidia-smi not found in PATH. Is the NVIDIA driver installed?')
        sys.exit(1)

    # Cache process names: nvidia-smi -q is slow but the names rarely
    # change for live PIDs — refresh this map every 6 cycles
    name_cache: Dict[int, str] = {}
    name_cache_age = 999

    try:
        while True:
            t0 = time.time()
            try:
                summary = query_gpu_summary(smi)
            except Exception as e:
                print(f"summary query failed: {e}", file=sys.stderr)
                time.sleep(args.interval)
                continue

            pmon = query_processes_pmon(smi)
            if name_cache_age >= 6 or not name_cache:
                name_cache = query_processes_full(smi)
                name_cache_age = 0
            else:
                name_cache_age += 1
            vram = {} if args.no_vram else query_vram_per_pid()

            # PIDs that show up in pmon/vram but lack a name from nvidia-smi
            # are usually system processes (dwm.exe, services). Resolve them
            # via PowerShell once and cache the result.
            unknown = [
                pid for pid in (set(pmon) | set(vram))
                if pid not in name_cache or not name_cache.get(pid)
                or name_cache[pid] == '?'
            ]
            if unknown and not args.no_vram:
                resolved = query_pid_names_via_powershell(unknown)
                name_cache.update(resolved)

            processes = merge_process_data(pmon, name_cache, vram)

            # ANSI clear screen + home (Windows Terminal supports this; on
            # the legacy console it'll just look like extra newlines)
            print('\033[2J\033[H', end='')
            now = time.strftime('%Y-%m-%d %H:%M:%S')
            elapsed = time.time() - t0
            print(f'{now}  (query took {elapsed:.1f}s)')
            print(render(summary, processes, args.top))

            sleep_for = max(0.0, args.interval - (time.time() - t0))
            time.sleep(sleep_for)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
