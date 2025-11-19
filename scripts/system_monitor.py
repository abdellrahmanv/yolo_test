import psutil
import subprocess

def get_temp():
    temp = subprocess.check_output(["vcgencmd", "measure_temp"]).decode()
    return float(temp.replace("temp=", "").replace("'C\n", ""))

def get_system_stats():
    return psutil.cpu_percent(), psutil.virtual_memory().percent, get_temp()
