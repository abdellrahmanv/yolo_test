import psutil
import subprocess

def get_temp():
    """Get Raspberry Pi CPU temperature"""
    try:
        temp = subprocess.check_output(["vcgencmd", "measure_temp"]).decode()
        return float(temp.replace("temp=", "").replace("'C\n", ""))
    except:
        return 0.0

def get_system_stats():
    """Get CPU%, RAM%, and Temperature"""
    return psutil.cpu_percent(), psutil.virtual_memory().percent, get_temp()
