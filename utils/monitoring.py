import psutil
import time
import torch
import os
from datetime import datetime

class ResourceMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.peak_memory = 0
        
    def get_stats(self):
        """Return current resource usage statistics"""
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        cpu_percent = process.cpu_percent(interval=1)
        
        # Update peak memory
        current_mem = mem_info.rss / 1024 / 1024  # MB
        self.peak_memory = max(self.peak_memory, current_mem)
        
        return {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "cpu_percent": cpu_percent,
            "memory_mb": current_mem,
            "peak_memory_mb": self.peak_memory,
            "elapsed_sec": round(time.time() - self.start_time, 1),
            "torch_threads": torch.get_num_threads()
        }

def initialize_monitoring():
    """Initialize global monitoring"""
    global monitor
    monitor = ResourceMonitor()

def print_system_stats(context=""):
    """Print current resource usage"""
    stats = monitor.get_stats()
    print(
        f"[{stats['timestamp']}] {context.ljust(15)} | "
        f"CPU: {stats['cpu_percent']:5.1f}% | "
        f"MEM: {stats['memory_mb']:6.1f}MB (Peak: {stats['peak_memory_mb']:6.1f}MB) | "
        f"Time: {stats['elapsed_sec']:5.1f}s"
    )

def log_resource_usage(phase_name):
    """Log resource usage to file"""
    stats = monitor.get_stats()
    log_entry = (
        f"{datetime.now().isoformat()},"
        f"{phase_name},"
        f"{stats['cpu_percent']},"
        f"{stats['memory_mb']},"
        f"{stats['peak_memory_mb']},"
        f"{stats['elapsed_sec']},"
        f"{stats['torch_threads']}\n"
    )
    
    with open("training_log.csv", "a") as f:
        if os.stat("training_log.csv").st_size == 0:
            f.write("timestamp,phase,cpu_percent,memory_mb,peak_memory_mb,elapsed_sec,threads\n")
        f.write(log_entry)
    
    print_system_stats(phase_name)