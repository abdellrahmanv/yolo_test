import csv
import os
from datetime import datetime

def generate_markdown_report(model_name, csv_path):
    """
    Generate a markdown report with ASCII graphs from benchmark data.
    """
    # Read CSV data
    frames = []
    fps_data = []
    cpu_data = []
    ram_data = []
    temp_data = []
    detections_data = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(int(row['frame']))
            fps_data.append(float(row['fps']))
            cpu_data.append(float(row['cpu']))
            ram_data.append(float(row['ram']))
            temp_data.append(float(row['temp']))
            detections_data.append(int(row['detections']))
    
    # Calculate statistics
    avg_fps = sum(fps_data) / len(fps_data)
    min_fps = min(fps_data)
    max_fps = max(fps_data)
    
    avg_cpu = sum(cpu_data) / len(cpu_data)
    avg_ram = sum(ram_data) / len(ram_data)
    avg_temp = sum(temp_data) / len(temp_data)
    max_temp = max(temp_data)
    
    total_detections = sum(detections_data)
    
    # Generate markdown report
    report_path = f"logs/{model_name}_report.md"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(report_path, 'w') as f:
        f.write(f"# 🚀 YOLO Benchmark Report: {model_name.upper()}\n\n")
        f.write(f"**Test Date:** {timestamp}\n\n")
        f.write(f"**Total Frames:** {len(frames)}\n\n")
        f.write(f"---\n\n")
        
        # Summary Statistics
        f.write(f"## 📊 Summary Statistics\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| **Average FPS** | {avg_fps:.2f} |\n")
        f.write(f"| **Min FPS** | {min_fps:.2f} |\n")
        f.write(f"| **Max FPS** | {max_fps:.2f} |\n")
        f.write(f"| **Average CPU** | {avg_cpu:.1f}% |\n")
        f.write(f"| **Average RAM** | {avg_ram:.1f}% |\n")
        f.write(f"| **Average Temp** | {avg_temp:.1f}°C |\n")
        f.write(f"| **Max Temp** | {max_temp:.1f}°C |\n")
        f.write(f"| **Total Detections** | {total_detections} |\n\n")
        
        f.write(f"---\n\n")
        
        # FPS Graph
        f.write(f"## 📈 FPS Over Time\n\n")
        f.write(f"```\n")
        f.write(generate_ascii_graph(fps_data, "FPS", height=10))
        f.write(f"```\n\n")
        
        # CPU Usage Graph
        f.write(f"## 🔥 CPU Usage Over Time\n\n")
        f.write(f"```\n")
        f.write(generate_ascii_graph(cpu_data, "CPU %", height=10))
        f.write(f"```\n\n")
        
        # RAM Usage Graph
        f.write(f"## 💾 RAM Usage Over Time\n\n")
        f.write(f"```\n")
        f.write(generate_ascii_graph(ram_data, "RAM %", height=10))
        f.write(f"```\n\n")
        
        # Temperature Graph
        f.write(f"## 🌡️ Temperature Over Time\n\n")
        f.write(f"```\n")
        f.write(generate_ascii_graph(temp_data, "Temp °C", height=10))
        f.write(f"```\n\n")
        
        # Detections Graph
        f.write(f"## 🎯 Detections Per Frame\n\n")
        f.write(f"```\n")
        f.write(generate_ascii_graph(detections_data, "Objects", height=10, is_integer=True))
        f.write(f"```\n\n")
        
        f.write(f"---\n\n")
        f.write(f"*Generated automatically by YOLO Benchmark Tool*\n")
    
    print(f"✅ Report generated: {report_path}")


def generate_ascii_graph(data, label, height=10, width=60, is_integer=False):
    """
    Generate an ASCII graph from data.
    """
    if not data:
        return "No data available\n"
    
    max_val = max(data)
    min_val = min(data)
    
    if max_val == min_val:
        max_val = min_val + 1
    
    graph = []
    
    # Header
    graph.append(f"{label}: Min={min_val:.1f}, Max={max_val:.1f}, Avg={sum(data)/len(data):.1f}\n")
    graph.append("\n")
    
    # Scale data to fit height
    scaled_data = []
    for val in data:
        scaled = int(((val - min_val) / (max_val - min_val)) * (height - 1))
        scaled_data.append(scaled)
    
    # Sample data if too many points
    if len(scaled_data) > width:
        step = len(scaled_data) / width
        sampled = [scaled_data[int(i * step)] for i in range(width)]
        scaled_data = sampled
    
    # Draw graph from top to bottom
    for row in range(height - 1, -1, -1):
        line = ""
        for val in scaled_data:
            if val >= row:
                line += "█"
            else:
                line += " "
        
        # Add axis label
        actual_val = min_val + (row / (height - 1)) * (max_val - min_val)
        if is_integer:
            graph.append(f"{int(actual_val):4d} |{line}\n")
        else:
            graph.append(f"{actual_val:5.1f}|{line}\n")
    
    # Bottom axis
    graph.append("     +" + "-" * len(scaled_data) + "\n")
    graph.append(f"      Frame: 0 → {len(data)}\n")
    
    return "".join(graph)


if __name__ == "__main__":
    # Test with sample data
    print("This module generates markdown reports from benchmark CSV files.")
