#!/usr/bin/env python3
"""
GPU monitoring utility for training progress
"""

import torch
import time
import psutil
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os

def monitor_gpu_usage(duration_minutes=5, interval_seconds=2):
    """Monitor GPU usage during training"""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    print(f"🖥️  Monitoring GPU usage for {duration_minutes} minutes...")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Interval: {interval_seconds} seconds")
    print("-" * 50)
    
    # Storage for metrics
    timestamps = []
    gpu_memory_used = []
    gpu_memory_total = []
    cpu_usage = []
    
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    
    try:
        while time.time() < end_time:
            current_time = datetime.now().strftime("%H:%M:%S")
            
            # GPU memory info
            gpu_mem_used = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            gpu_mem_percent = (gpu_mem_used / gpu_mem_total) * 100
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Store metrics
            timestamps.append(current_time)
            gpu_memory_used.append(gpu_mem_used)
            gpu_memory_total.append(gpu_mem_total)
            cpu_usage.append(cpu_percent)
            
            print(f"{current_time} | GPU Memory: {gpu_mem_used:.2f}/{gpu_mem_total:.2f} GB ({gpu_mem_percent:.1f}%) | CPU: {cpu_percent:.1f}%")
            
            time.sleep(interval_seconds)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    
    # Save and plot results
    if timestamps:
        save_monitoring_results(timestamps, gpu_memory_used, gpu_memory_total, cpu_usage)

def save_monitoring_results(timestamps, gpu_memory_used, gpu_memory_total, cpu_usage):
    """Save monitoring results to file and create plots"""
    
    # Create monitoring directory
    monitor_dir = "monitoring"
    os.makedirs(monitor_dir, exist_ok=True)
    
    # Save raw data
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_file = os.path.join(monitor_dir, f"gpu_monitoring_{timestamp_str}.json")
    
    monitoring_data = {
        'timestamps': timestamps,
        'gpu_memory_used_gb': gpu_memory_used,
        'gpu_memory_total_gb': gpu_memory_total[0] if gpu_memory_total else 0,
        'cpu_usage_percent': cpu_usage,
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"
    }
    
    with open(data_file, 'w') as f:
        json.dump(monitoring_data, f, indent=2)
    
    print(f"📊 Monitoring data saved to: {data_file}")
    
    # Create plots
    plot_file = os.path.join(monitor_dir, f"gpu_monitoring_{timestamp_str}.png")
    create_monitoring_plots(timestamps, gpu_memory_used, gpu_memory_total, cpu_usage, plot_file)

def create_monitoring_plots(timestamps, gpu_memory_used, gpu_memory_total, cpu_usage, save_path):
    """Create monitoring plots"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('GPU Training Monitoring', fontsize=16)
    
    # GPU Memory Usage
    gpu_total = gpu_memory_total[0] if gpu_memory_total else 16  # Default to 16GB
    ax1.plot(range(len(timestamps)), gpu_memory_used, 'b-', label='Used Memory', linewidth=2)
    ax1.axhline(y=gpu_total, color='r', linestyle='--', label=f'Total Memory ({gpu_total:.1f} GB)')
    ax1.fill_between(range(len(timestamps)), gpu_memory_used, alpha=0.3)
    ax1.set_ylabel('GPU Memory (GB)')
    ax1.set_title(f'GPU Memory Usage - {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # CPU Usage
    ax2.plot(range(len(timestamps)), cpu_usage, 'g-', label='CPU Usage', linewidth=2)
    ax2.fill_between(range(len(timestamps)), cpu_usage, alpha=0.3, color='green')
    ax2.set_ylabel('CPU Usage (%)')
    ax2.set_xlabel('Time Points')
    ax2.set_title('CPU Usage During Training')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Set x-axis labels (show every 5th timestamp)
    step = max(1, len(timestamps) // 10)
    x_ticks = range(0, len(timestamps), step)
    x_labels = [timestamps[i] for i in x_ticks]
    
    for ax in [ax1, ax2]:
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📈 Monitoring plots saved to: {save_path}")
    
    # Show summary statistics
    print(f"\n📈 Monitoring Summary:")
    print(f"   Average GPU Memory Usage: {sum(gpu_memory_used)/len(gpu_memory_used):.2f} GB")
    print(f"   Peak GPU Memory Usage: {max(gpu_memory_used):.2f} GB")
    print(f"   Average CPU Usage: {sum(cpu_usage)/len(cpu_usage):.1f}%")
    print(f"   Peak CPU Usage: {max(cpu_usage):.1f}%")

def get_current_gpu_status():
    """Get current GPU status"""
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    print("🔍 Current GPU Status:")
    print(f"   Device: {torch.cuda.get_device_name(0)}")
    print(f"   Memory Used: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"   Memory Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    print(f"   Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"   Memory Usage: {(torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory) * 100:.1f}%")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor GPU usage during training")
    parser.add_argument("--duration", type=int, default=5, help="Monitoring duration in minutes")
    parser.add_argument("--interval", type=int, default=2, help="Monitoring interval in seconds")
    parser.add_argument("--status", action="store_true", help="Show current GPU status only")
    
    args = parser.parse_args()
    
    if args.status:
        get_current_gpu_status()
    else:
        monitor_gpu_usage(args.duration, args.interval)
