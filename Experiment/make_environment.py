import sys
# 将项目根目录添加到系统路径，确保其他模块可以被正确导入
sys.path.append(r"/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/")

from typing import Optional, List, Tuple
import numpy as np
# 从环境模块导入距离矩阵初始化、空间大小定义函数
from Environment.environment import init_distance_matrix_and_radio_coverage_matrix, define_size_of_spaces
# 导入不同类型的车联网环境类（凸优化资源分配、随机资源分配、本地处理、边缘卸载载等场景）
from Environment.environment import vehicularNetworkEnv as ConvexResourceAllocationEnv
from Environment.environment_random_action import vehicularNetworkEnv as RandomResourceAllocationEnv
from Environment.environment_local_processing import vehicularNetworkEnv as LocalOffloadingEnv
from Environment.environment_offloaded_other_edge_nodes import vehicularNetworkEnv as EdgeOffloadEnv
from Environment.environment_old import vehicularNetworkEnv as OldEnv
from Environment.environment_global_actions import vehicularNetworkEnv as GlobalActionEnv
# 导入环境配置类
from Environment.environmentConfig import vehicularNetworkEnvConfig
# 导入数据结构类（车辆列表、时间片、任务列表、边缘节点列表）
from Environment.dataStruct import vehicleList, timeSlots, taskList, edgeList
# 导入文件操作工具（保存对象、初始化文件名）
from Utilities.FileOperator import save_obj, init_file_name

def get_default_environment(
        flatten_space: Optional[bool] = False,# 是否将环境空间展平（用于不同强化学习框架兼容）
        occuiped: Optional[bool] = False,# 是否考虑资源占用状态
        for_mad5pg: Optional[bool] = True,# 是否为MAD5PG算法适配环境
    ):
    # 初始化环境配置实例
    environment_config = vehicularNetworkEnvConfig(
        task_request_rate=0.7,# 任务请求率（车辆生成任务的概率）
    )
    # 为每个车辆分配独立的随机种子（确保仿真可复现）
    environment_config.vehicle_seeds += [i for i in range(environment_config.vehicle_number)]
    # 初始化时间片对象（定义仿真的时间范围和步长）
    time_slots= timeSlots(
        start=environment_config.time_slot_start, # 起始时间
        end=environment_config.time_slot_end, # 结束时间
        slot_length=environment_config.time_slot_length,# 每个时间片的长度（秒）
    )
    # 初始化任务列表（生成仿真所需的任务参数）
    task_list = taskList(
        tasks_number=environment_config.task_number,# 任务总数
        minimum_data_size=environment_config.task_minimum_data_size,# 任务数据量最小值
        maximum_data_size=environment_config.task_maximum_data_size,# 任务数据量最大值
        minimum_computation_cycles=environment_config.task_minimum_computation_cycles, # 任务计算周期最小值
        maximum_computation_cycles=environment_config.task_maximum_computation_cycles,# 任务计算周期最大值
        minimum_delay_thresholds=environment_config.task_minimum_delay_thresholds, # 任务延迟阈值最小值
        maximum_delay_thresholds=environment_config.task_maximum_delay_thresholds,# 任务延迟阈值最大值
        seed=environment_config.task_seed,# 任务生成随机种子
    )
    # 初始化车辆列表（加载车辆轨迹并生成车辆相关参数）
    vehicle_list = vehicleList(
        edge_number=environment_config.edge_number,# 边缘节点数量
        communication_range=environment_config.communication_range,# 通信范围（米）
        vehicle_number=environment_config.vehicle_number,# 车辆总数
        time_slots=time_slots,# 时间片配置
        trajectories_file_name=environment_config.trajectories_file_name,# 车辆轨迹数据文件   处理后的CSV文件
        slot_number=environment_config.time_slot_number,# 时间片总数
        task_number=environment_config.task_number, # 任务总数
        task_request_rate=environment_config.task_request_rate, # 任务请求率
        seeds=environment_config.vehicle_seeds,# 车辆随机种子列表
    )
    
    # print("len(vehicle_list): ", len(vehicle_list.get_vehicle_list()))
    # print("vehicle_number: ", environment_config.vehicle_number)

    # 初始化边缘节点列表（定义边缘节点的位置、计算资源等参数）
    edge_list = edgeList(
        edge_number=environment_config.edge_number,# 边缘节点数量
        power=environment_config.edge_power,# 边缘节点发射功率
        bandwidth=environment_config.edge_bandwidth,# 边缘节点带宽
        minimum_computing_cycles=environment_config.edge_minimum_computing_cycles,# 最小计算周期
        maximum_computing_cycles=environment_config.edge_maximum_computing_cycles,# 最大计算周期
        communication_range=environment_config.communication_range,# 通信范围
        # 边缘节点的坐标（3x3网格分布，覆盖3000x3000米区域）
        edge_xs=[500, 1500, 2500, 500, 1500, 2500, 500, 1500, 2500],
        edge_ys=[2500, 2500, 2500, 1500, 1500, 1500, 500, 500, 500],
        seed=environment_config.edge_seed,# 边缘节点随机种子
    )
    # 初始化距离矩阵（车辆-边缘节点距离）、信道条件矩阵、车辆-边缘节点关联索引
    distance_matrix, channel_condition_matrix, vehicle_index_within_edges, vehicle_observed_index_within_edges = init_distance_matrix_and_radio_coverage_matrix(env_config=environment_config, vehicle_list=vehicle_list, edge_list=edge_list)
    # 定义动作空间、观测空间、奖励空间、 Critic网络动作空间的大小
    environment_config.vehicle_number_within_edges = int(environment_config.vehicle_number / environment_config.edge_number)
    environment_config.action_size, environment_config.observation_size, environment_config.reward_size, \
            environment_config.critic_network_action_size = define_size_of_spaces(vehicle_number_within_edges=environment_config.vehicle_number_within_edges, edge_number=environment_config.edge_number, task_assigned_number=environment_config.task_assigned_number)
    
    print("environment_config.action_size: ", environment_config.action_size)
    print("environment_config.observation_size: ", environment_config.observation_size)
    print("environment_config.reward_size: ", environment_config.reward_size)
    print("environment_config.critic_network_action_size: ", environment_config.critic_network_action_size)

    # 以下为不同类型环境的初始化代码

    # 1. 凸优化资源分配环境（基于凸优化的资源分配策略）
    # convexEnvironment = ConvexResourceAllocationEnv(
    #     envConfig = environment_config,
    #     time_slots = time_slots,
    #     task_list = task_list,
    #     vehicle_list = vehicle_list,
    #     edge_list = edge_list,
    #     distance_matrix = distance_matrix, 
    #     channel_condition_matrix = channel_condition_matrix, 
    #     vehicle_index_within_edges = vehicle_index_within_edges,
    #     vehicle_observed_index_within_edges = vehicle_observed_index_within_edges,
    #     flatten_space = flatten_space,
    #     occuiped = occuiped,
    #     for_mad5pg = for_mad5pg, 
    # )

    # 2. 随机资源分配环境（随机分配资源的基准算法）
    # randomEnvironment = RandomResourceAllocationEnv(
    #     envConfig = environment_config,
    #     time_slots = time_slots,
    #     task_list = task_list,
    #     vehicle_list = vehicle_list,
    #     edge_list = edge_list,
    #     distance_matrix = distance_matrix, 
    #     channel_condition_matrix = channel_condition_matrix, 
    #     vehicle_index_within_edges = vehicle_index_within_edges,
    #     vehicle_observed_index_within_edges = vehicle_observed_index_within_edges,
    #     flatten_space = flatten_space,
    #     occuiped = occuiped,
    #     for_mad5pg = for_mad5pg, 
    # )

    # 3. 本地处理环境（任务仅在本地车辆处理的基准算法）
    # localEnvironment = LocalOffloadingEnv(
    #     envConfig = environment_config,
    #     time_slots = time_slots,
    #     task_list = task_list,
    #     vehicle_list = vehicle_list,
    #     edge_list = edge_list,
    #     distance_matrix = distance_matrix, 
    #     channel_condition_matrix = channel_condition_matrix, 
    #     vehicle_index_within_edges = vehicle_index_within_edges,
    #     vehicle_observed_index_within_edges = vehicle_observed_index_within_edges,
    #     flatten_space = flatten_space,
    #     occuiped = occuiped,
    #     for_mad5pg = for_mad5pg, 
    # )

    # 4. 边缘卸载环境（任务仅卸载到边缘节点的基准算法）
    # edgeEnvironment = EdgeOffloadEnv(
    #     envConfig = environment_config,
    #     time_slots = time_slots,
    #     task_list = task_list,
    #     vehicle_list = vehicle_list,
    #     edge_list = edge_list,
    #     distance_matrix = distance_matrix, 
    #     channel_condition_matrix = channel_condition_matrix, 
    #     vehicle_index_within_edges = vehicle_index_within_edges,
    #     vehicle_observed_index_within_edges = vehicle_observed_index_within_edges,
    #     flatten_space = flatten_space,
    #     occuiped = occuiped,
    #     for_mad5pg = for_mad5pg, 
    # )

    # 5. 旧版本环境（可能为早期实现的环境，用于对比）
    # oldEnvironment = OldEnv(
    #             envConfig = environment_config,
    #     time_slots = time_slots,
    #     task_list = task_list,
    #     vehicle_list = vehicle_list,
    #     edge_list = edge_list,
    #     distance_matrix = distance_matrix, 
    #     channel_condition_matrix = channel_condition_matrix, 
    #     vehicle_index_within_edges = vehicle_index_within_edges,
    #     vehicle_observed_index_within_edges = vehicle_observed_index_within_edges,
    #     flatten_space = flatten_space,
    #     occuiped = occuiped,
    #     for_mad5pg = for_mad5pg, 
    # )

    # 6. 全局动作环境（当前启用的环境，支持全局动作决策）
    globalActionEnv = GlobalActionEnv(
        envConfig = environment_config,
        time_slots = time_slots,
        task_list = task_list,
        vehicle_list = vehicle_list,
        edge_list = edge_list,
        distance_matrix = distance_matrix, 
        channel_condition_matrix = channel_condition_matrix, 
        vehicle_index_within_edges = vehicle_index_within_edges,
        vehicle_observed_index_within_edges = vehicle_observed_index_within_edges,
        flatten_space = flatten_space,
        occuiped = occuiped,
        for_mad5pg = for_mad5pg, 
    )
    # 初始化环境数据文件的保存路径和名称
    file_name = init_file_name()
    # 保存不同环境的实例到.pkl文件（当前仅保存全局动作环境）
    # save_obj(randomEnvironment, file_name["random_environment_name"])
    # save_obj(convexEnvironment, file_name["convex_environment_name"])
    # save_obj(localEnvironment, file_name["local_environment_name"])
    # save_obj(edgeEnvironment, file_name["edge_environment_name"])
    # save_obj(oldEnvironment, file_name["old_environment_name"])
    save_obj(globalActionEnv, file_name["global_environment_name"])

if __name__ == "__main__":
    # for d4pg
    get_default_environment(flatten_space=True)
    # for mad4pg
    # get_default_environment(for_mad5pg=True)