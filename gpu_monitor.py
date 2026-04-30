#!/usr/bin/env python3
"""
Live GPU usage monitor — wraps `nvidia-smi pmon` for a less-cluttered view.

Shows per-process GPU compute % and memory MB, refreshed every interval.
Highlights the heavy hitters so you can see at a glance whether training
is hogging the card or other processes are competing.

Usage:
    python gpu_monitor.py                # 2-second refresh
    python gpu_monitor.py --interval 5   # 5-second refresh
    python gpu_monitor.py --top 5        # show top 5 processes only
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


def query_gpu_summary(smi: str) -> Dict:
    """Return dict with util_pct, mem_used_mib, mem_total_mib, power_w."""
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


def query_processes(smi: str) -> List[Dict]:
    """Return list of {pid, name, sm_pct, mem_mib} for compute+graphics procs.

    Uses `nvidia-smi pmon -c 1`. The `pmon` view gives per-process compute %
    which is what we actually want for diagnosing UI lockups."""
    try:
        out = subprocess.check_output(
            [smi, 'pmon', '-c', '1', '-s', 'um'],
            text=True, stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return []

    procs: List[Dict] = []
    for line in out.splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        parts = line.split()
        # pmon columns: gpu pid type sm mem enc dec command  (fixed in modern smi)
        # Schema can vary; key columns we want: pid, sm, mem (MB), command.
        try:
            pid = int(parts[1])
        except (IndexError, ValueError):
            continue
        ptype = parts[2] if len(parts) > 2 else '?'
        # SM and MEM columns can be '-' if process isn't currently active
        def to_int(s: str) -> int:
            try:
                return int(s)
            except ValueError:
                return 0
        sm = to_int(parts[3]) if len(parts) > 3 else 0
        mem = to_int(parts[4]) if len(parts) > 4 else 0
        # command is the last field; on Windows it's the exe basename
        name = parts[-1] if len(parts) >= 6 else '?'
        procs.append({
            'pid': pid, 'name': name, 'type': ptype,
            'sm_pct': sm, 'mem_mib': mem,
        })
    return procs


def fallback_processes(smi: str) -> List[Dict]:
    """When pmon doesn't expose sm%, fall back to memory-only listing via
    --query-compute-apps and --query-accounted-apps."""
    procs: List[Dict] = []
    try:
        out = subprocess.check_output(
            [smi, '--query-compute-apps=pid,used_memory,name',
             '--format=csv,noheader,nounits'],
            text=True, stderr=subprocess.DEVNULL,
        )
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            pid_s, mem_s, name = (s.strip() for s in line.split(',', 2))
            procs.append({
                'pid': int(pid_s),
                'name': os.path.basename(name) or name,
                'type': 'C',
                'sm_pct': 0,
                'mem_mib': int(mem_s),
            })
    except Exception:
        pass
    return procs


def render(summary: Dict, procs: List[Dict], top_n: int) -> str:
    procs_sorted = sorted(procs,
                          key=lambda p: (p['sm_pct'], p['mem_mib']),
                          reverse=True)[:top_n]

    lines = []
    util_bar = '#' * (summary['util_pct'] // 5)
    util_bar = util_bar.ljust(20)
    mem_pct = 100.0 * summary['mem_used_mib'] / max(1, summary['mem_total_mib'])
    lines.append(
        f"GPU util: {summary['util_pct']:>3d}% [{util_bar}] | "
        f"VRAM: {summary['mem_used_mib']:>5d}/{summary['mem_total_mib']:>5d} MB "
        f"({mem_pct:>4.1f}%) | "
        f"{summary['power_w']:>5.1f} W | {summary['temp_c']:>3d}°C")
    lines.append('')
    if not procs_sorted:
        lines.append('  (no compute/graphics processes reported by pmon — '
                     'use Task Manager → Performance → GPU for full breakdown)')
    else:
        lines.append(f"  {'PID':>7}  {'TYPE':<4}  {'SM%':>4}  {'MEM(MB)':>8}  PROCESS")
        for p in procs_sorted:
            mark = '>>' if p['sm_pct'] >= 50 else '  '
            lines.append(
                f"{mark}{p['pid']:>7}  {p['type']:<4}  "
                f"{p['sm_pct']:>3d}%  {p['mem_mib']:>8d}  {p['name']}")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Live GPU process monitor')
    parser.add_argument('--interval', type=float, default=2.0,
                        help='Refresh interval in seconds (default: 2)')
    parser.add_argument('--top', type=int, default=8,
                        help='Show top N processes (default: 8)')
    args = parser.parse_args()

    smi = nvidia_smi_path()
    if smi is None:
        print('nvidia-smi not found in PATH. Is the NVIDIA driver installed?')
        sys.exit(1)

    try:
        while True:
            try:
                summary = query_gpu_summary(smi)
            except Exception as e:
                print(f"summary query failed: {e}", file=sys.stderr)
                time.sleep(args.interval)
                continue
            procs = query_processes(smi) or fallback_processes(smi)
            # Clear screen (ANSI; Windows Terminal supports this; falls back
            # to printing newlines on basic consoles)
            print('\033[2J\033[H', end='')
            print(time.strftime('%Y-%m-%d %H:%M:%S'))
            print(render(summary, procs, args.top))
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
