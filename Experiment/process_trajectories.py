import sys
# 将项目根目录添加到系统路径，确保其他模块（如Environment.utilities）可以被正确导入
# sys.path.append(r"/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/")
sys.path.append(r"D:\lxl\2025\code\origin_code\Game-Theoretic-Deep-Reinforcement-Learning-main\\")
from Environment.utilities import vehicleTrajectoriesProcessor

if __name__ == "__main__":
    """Vehicle Trajectories Processor related."""
    """车辆轨迹轨迹处理相关的主程序入口"""
    # 输入的原始轨迹轨迹CSV文件路径（来自滴滴GAIA开放数据集）
    trajectories_file_name: str = 'CSV/gps_20161116'
    longitude_min: float = 104.04565967220308 # 感兴趣区域的最小经度（地理边界）
    latitude_min: float = 30.654605745741608# 感兴趣区域的最小纬度（地理边界）
    trajectories_time_start: str = '2016-11-16 23:00:00'# 轨迹数据的时间范围起点
    trajectories_time_end: str = '2016-11-16 23:05:00'# 轨迹数据的时间范围终点
    trajectories_out_file_name: str = 'CSV/trajectories_20161116_2300_2305'# 处理后的轨迹数据输出文件路径
    edge_number: int = 9
    communication_range: float = 500
    
    processor = vehicleTrajectoriesProcessor(
        file_name=trajectories_file_name, 
        longitude_min=longitude_min, 
        latitude_min=latitude_min,
        edge_number=edge_number,
        map_width=3000.0,
        communication_range=communication_range,
        time_start=trajectories_time_start,
        time_end=trajectories_time_end, 
        out_file=trajectories_out_file_name,
    )    
    