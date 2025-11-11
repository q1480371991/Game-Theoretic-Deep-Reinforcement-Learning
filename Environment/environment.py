
"""Vehicular Network Environments."""

# ---------------------------------------------------------------------------
# 车联网环境的核心实现。强化学习智能体通过 `dm_env.Environment` 接口与该环境交互，
# 获取观测、执行动作并得到奖励。本文件新增了详细的中文注释，用于解释每一个步骤
# 所体现的物理意义与实现思路，帮助研究者快速了解仿真流程。
# ---------------------------------------------------------------------------

import time
import dm_env
from dm_env import specs
from acme.types import NestedSpec
import numpy as np
# typing 用于类型注解，帮助理解各函数输入输出的数据结构。
from typing import List, Tuple, NamedTuple, Optional

# 环境配置与核心数据结构，负责生成车辆、任务、边缘节点等基础信息。
import Environment.environmentConfig as env_config
# timeSlots、taskList、edgeList、vehicleList 分别封装了时间轴、任务参数、边缘服务器以及车辆的动态信息。
from Environment.dataStruct import timeSlots, taskList, edgeList, vehicleList
# 通信层相关的工具函数，用于计算信道增益、SINR、传输速率等关键指标。
from Environment.utilities import compute_channel_gain, generate_complex_normal_distribution, compute_transmission_rate, compute_SINR, cover_mW_to_W
np.set_printoptions(threshold=np.inf)
from Log.logger import myapp
# 定义了 vehicularNetworkEnvConfig 数据类（基于 dataclasses），集中存储环境的所有配置参数。
# 模拟车联网环境的动态变化（如车辆移动、任务生成、资源分配），为强化学习智能体提供交互接口（状态观测、动作反馈、奖励计算）。
# ===========================================================================
# vehicularNetworkEnv: 继承自 dm_env.Environment 的车联网环境，实现强化学习所需的交互接口。
# ===========================================================================
class vehicularNetworkEnv(dm_env.Environment):
    """Vehicular Network Environment built on the dm_env framework."""
    
    def __init__(
        self, 
        envConfig: Optional[env_config.vehicularNetworkEnvConfig] = None,
        time_slots: Optional[timeSlots] = None,
        task_list: Optional[taskList] = None,
        vehicle_list: Optional[vehicleList] = None,
        edge_list: Optional[edgeList] = None,
        distance_matrix: Optional[np.ndarray] = None, 
        channel_condition_matrix: Optional[np.ndarray] = None, 
        vehicle_index_within_edges: Optional[List[List[List[int]]]] = None,
        vehicle_observed_index_within_edges: Optional[List[List[List[int]]]] = None,
        flatten_space: Optional[bool] = False,
        occuiped: Optional[bool] = False,
        for_mad5pg: Optional[bool] = True,
    ) -> None:
        """Initialize the environment."""
        # 初始化流程：读取配置、构建基础对象、预计算距离矩阵，并重置资源占用状态。
        # 根据调用者传入的可选参数可以复用已有的车辆轨迹、任务列表或信道状态。
        # 未提供配置时使用默认参数，并补充车辆的随机种子以保证可重复性。
        if envConfig is None:
            self._config = env_config.vehicularNetworkEnvConfig()
            self._config.vehicle_seeds += [i for i in range(self._config.vehicle_number)]
            self._config.vehicle_number_within_edges = int(self._config.vehicle_number / self._config.edge_number)
            self._config.action_size, self._config.observation_size, self._config.reward_size, \
                self._config.critic_network_action_size = define_size_of_spaces(self._config.vehicle_number_within_edges, self._config.edge_number, self._config.task_assigned_number)
        else:
            self._config = envConfig
        
        # 初始化距离矩阵和信道条件矩阵，判断车辆是否在边缘节点覆盖范围内。
        if distance_matrix is None:
            self._distance_matrix, self._channel_condition_matrix, self._vehicle_index_within_edges, self._vehicle_observed_index_within_edges = init_distance_matrix_and_radio_coverage_matrix(
                env_config=self._config,
                vehicle_list=vehicle_list,
                edge_list=edge_list,
            )
        else:
            self._distance_matrix = distance_matrix
            self._channel_condition_matrix = channel_condition_matrix
            self._vehicle_index_within_edges = vehicle_index_within_edges
            self._vehicle_observed_index_within_edges = vehicle_observed_index_within_edges
        
        # 构建时间片对象，控制环境推进节奏。
        if time_slots is None:
            self._time_slots: timeSlots = timeSlots(
                start=self._config.time_slot_start,
                end=self._config.time_slot_end,
                slot_length=self._config.time_slot_length,
            )
        else:
            self._time_slots = time_slots
        # 生成任务列表，随机化任务数据量、计算量和时延阈值。
        if task_list is None:
            self._task_list: taskList = taskList(
                tasks_number=self._config.task_number,
                minimum_data_size=self._config.task_minimum_data_size,
                maximum_data_size=self._config.task_maximum_data_size,
                minimum_computation_cycles=self._config.task_minimum_computation_cycles,
                maximum_computation_cycles=self._config.task_maximum_computation_cycles,
                minimum_delay_thresholds=self._config.task_minimum_delay_thresholds,
                maximum_delay_thresholds=self._config.task_maximum_delay_thresholds,
                seed=self._config.task_seed,
            )
        else:
            self._task_list = task_list
        # 构造车辆集合，包含车辆轨迹和任务请求到达过程。
        if vehicle_list is None:
            self._vehicle_list: vehicleList = vehicleList(
                vehicle_number=self._config.vehicle_number,
                time_slots=self._time_slots,
                trajectories_file_name=self._config.trajectories_file_name,
                slot_number=self._config.time_slot_number,
                task_number=self._config.task_number,
                task_request_rate=self._config.task_request_rate,
                seeds=self._config.vehicle_seeds,
            )
        else:
            self._vehicle_list = vehicle_list
        # 初始化边缘节点属性（发射功率、带宽、算力、通信覆盖半径等）。
        if edge_list is None:
            self._edge_list: edgeList = edgeList(
                edge_number=self._config.edge_number,
                power=self._config.edge_power,
                bandwidth=self._config.edge_bandwidth,
                minimum_computing_cycles=self._config.edge_minimum_computing_cycles,
                maximum_computing_cycles=self._config.edge_maximum_computing_cycles,
                communication_range=self._config.communication_range,
                edge_xs=[500, 1500, 2500, 500, 1500, 2500, 500, 1500, 2500],
                edge_ys=[2500, 2500, 2500, 1500, 1500, 1500, 500, 500, 500],
                seed=self._config.edge_seed,
            )
        else:
            self._edge_list = edge_list
                    
        self._reward: np.ndarray = np.zeros(self._config.reward_size)
        
        # 资源占用矩阵用于模拟跨时间片的资源预留，使后续时刻可见已被占用的功率和算力。
        self._occupied_power = np.zeros(shape=(self._config.edge_number, self._config.time_slot_number))
        self._occupied_computing_resources = np.zeros(shape=(self._config.edge_number, self._config.time_slot_number))
        
        # _reset_next_step 标志在终止时间片后触发重置逻辑。
        self._reset_next_step: bool = True
        # _flatten_space 控制是否将多维观测/动作展平以适配特定算法。
        # _occuiped 表示是否考虑资源占用约束，_for_mad5pg 控制是否输出 MAD5PG 算法特定的特征。
        self._flatten_space: bool = flatten_space
        self._occuiped: bool = occuiped
        self._for_mad5pg: bool = for_mad5pg
        
    # ----------------------------- 环境交互接口 -----------------------------
    def reset(self) -> dm_env.TimeStep:
        """Resets the environment, clears resource occupancy and returns the first TimeStep."""
        # 回合开始时清空时间片和资源占用，返回初始观测。
        self._time_slots.reset()
        self._occupied_power = np.zeros(shape=(self._config.edge_number, self._config.time_slot_number))
        self._occupied_computing_resources = np.zeros(shape=(self._config.edge_number, self._config.time_slot_number))
        
        self._reset_next_step = False
        
        return dm_env.restart(observation=self._observation())

    def step(self, action: np.ndarray):
        # 根据智能体给定的动作计算奖励、推进时间片，并输出新的观测。
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        """        
        if self._reset_next_step:
            return self.reset()
        # 调用核心的 convex optimization 奖励计算，得到即时奖励及统计信息。
        self._reward, cumulative_reward, average_vehicle_SINR, average_vehicle_intar_interference, average_vehicle_inter_interference, \
            average_vehicle_interference, average_transmision_time, average_wired_transmission_time, average_execution_time, average_service_time, successful_serviced_number, task_offloaded_number, task_required_number = self.compute_reward_with_convex_optimization(action)
        # print("compute_reward time taken: ", time.time() - time_start)
        
        # 记录耗时以便调试性能瓶颈。
        time_start = time.time()
        observation = self._observation()
        # print("observation time taken: ", time.time() - time_start)
        # check for termination
        # 若时间片达到终点，则返回终止状态并在下一步重置环境。
        if self._time_slots.is_end():
            self._reset_next_step = True
            return dm_env.termination(observation=observation, reward=self._reward), cumulative_reward, average_vehicle_SINR, average_vehicle_intar_interference, average_vehicle_inter_interference, \
            average_vehicle_interference, average_transmision_time, average_wired_transmission_time, average_execution_time, average_service_time, successful_serviced_number, task_offloaded_number, task_required_number
        # 尚未结束时推进时间片，继续下一次交互。
        self._time_slots.add_time()
        return dm_env.transition(observation=observation, reward=self._reward), cumulative_reward, average_vehicle_SINR, average_vehicle_intar_interference, average_vehicle_inter_interference, \
            average_vehicle_interference, average_transmision_time, average_wired_transmission_time, average_execution_time, average_service_time, successful_serviced_number, task_offloaded_number, task_required_number
    
    # ------------------------- 资源分配子过程 -------------------------
    def get_transmission_power_with_convex_optimization(self):
        """通过分布式凸优化方案为每个车辆分配发射功率。"""
        
        # 以下矩阵记录每辆车在各个候选边缘节点下的信道质量与干扰情况。
        vehicle_SINR = np.zeros((self._config.vehicle_number, self._config.edge_number + 1))
        
        vehicle_intar_edge_inference = np.zeros((self._config.vehicle_number, self._config.edge_number + 1))
        vehicle_inter_edge_inference = np.zeros((self._config.vehicle_number, self._config.edge_number + 1))
        
        vehicle_edge_transmission_power = np.zeros((self._config.vehicle_number, self._config.edge_number))
        
        # 遍历每个边缘节点，统计覆盖范围内的车辆并均衡初始功率。
        for edge_index in range(self._config.edge_number):
            vehicle_index_within_edge = self._vehicle_index_within_edges[edge_index][self._time_slots.now()]
            tasks_number_within_edge = len(vehicle_index_within_edge)
            the_edge = self._edge_list.get_edge_by_index(edge_index)
            
            edge_power = the_edge.get_power()
            edge_occupied_power = self._occupied_power[edge_index][self._time_slots.now()]
            for i in range(int(tasks_number_within_edge)):
                vehicle_index = vehicle_index_within_edge[i]
                # 在考虑资源占用的设置下，需要将后续时间片的资源提前扣除。
                if self._occuiped:
                    if edge_power - edge_occupied_power <= 0:
                        vehicle_edge_transmission_power[vehicle_index][edge_index] = 0
                    else:
                        vehicle_edge_transmission_power[vehicle_index][edge_index] = 1 / tasks_number_within_edge * (edge_power - edge_occupied_power)
                else:
                    vehicle_edge_transmission_power[vehicle_index][edge_index] = 1 /tasks_number_within_edge * edge_power
                        
        
        # 依据初始化的功率估算同一边缘节点内/不同节点之间的干扰量。
        """Compute the inference"""
        for edge_index in range(self._config.edge_number):
            
            vehicle_index_within_edge = self._vehicle_index_within_edges[edge_index][self._time_slots.now()]
            
            edge_inter_interference = np.zeros((self._config.edge_number))
            
            for other_edge_index in range(self._config.edge_number):
                if other_edge_index != edge_index:
                    vehicle_index_within_other_edge = self._vehicle_index_within_edges[other_edge_index][self._time_slots.now()]
                    for other_vehicle_index in vehicle_index_within_other_edge:
                        other_channel_condition = self._channel_condition_matrix[other_vehicle_index][edge_index][self._time_slots.now()]
                        inter_interference = np.power(np.absolute(other_channel_condition), 2) * cover_mW_to_W(vehicle_edge_transmission_power[other_vehicle_index][other_edge_index])
                        edge_inter_interference[other_edge_index] += inter_interference
            
            if vehicle_index_within_edge != []:
                for vehicle_index in vehicle_index_within_edge:
                    channel_condition = self._channel_condition_matrix[vehicle_index][edge_index][self._time_slots.now()]
                    for other_vehicle_index in vehicle_index_within_edge:
                        if other_vehicle_index != vehicle_index:
                            other_channel_condition = self._channel_condition_matrix[other_vehicle_index][edge_index][self._time_slots.now()]
                            if other_channel_condition < channel_condition:
                                vehicle_intar_edge_inference[vehicle_index, -1] += np.power(np.absolute(other_channel_condition), 2) * cover_mW_to_W(vehicle_edge_transmission_power[other_vehicle_index][edge_index])
                            
        # 根据干扰值与信道增益计算各车辆的信噪比，后续用于自适应调整功率。
        """Compute the SINR"""
        for edge_index in range(self._config.edge_number):
            if self._vehicle_index_within_edges[edge_index][self._time_slots.now()] != []:
                for vehicle_index in self._vehicle_index_within_edges[edge_index][self._time_slots.now()]:
                    vehicle_SINR[vehicle_index, -1] = compute_SINR(
                        white_gaussian_noise=self._config.white_gaussian_noise, 
                        channel_condition=self._channel_condition_matrix[vehicle_index][edge_index][self._time_slots.now()],
                        transmission_power=vehicle_edge_transmission_power[vehicle_index][edge_index],
                        intra_edge_interference=vehicle_intar_edge_inference[vehicle_index][-1],
                        inter_edge_interference=vehicle_inter_edge_inference[vehicle_index][-1],)
        
        # 重新分配功率：SINR 越高的车辆分配更多功率，以最大化传输效率。
        """Updata the transmission power"""
        
        vehicle_edge_transmission_power = np.zeros((self._config.vehicle_number, self._config.edge_number))

        for edge_index in range(self._config.edge_number):
            vehicle_index_within_edge = self._vehicle_index_within_edges[edge_index][self._time_slots.now()]
            tasks_number_within_edge = len(vehicle_index_within_edge)
            the_edge = self._edge_list.get_edge_by_index(edge_index)
            
            edge_power = the_edge.get_power()
            edge_occupied_power = self._occupied_power[edge_index][self._time_slots.now()]
            
            divided_sum = 0
            for i in range(int(tasks_number_within_edge)):
                vehicle_index = vehicle_index_within_edge[i]
                divided_sum = divided_sum + ( vehicle_SINR[vehicle_index, -1] / (1 + vehicle_SINR[vehicle_index, -1]))
            
            for i in range(int(tasks_number_within_edge)):
                vehicle_index = vehicle_index_within_edge[i]
                if self._occuiped:
                    if edge_power - edge_occupied_power <= 0:
                        vehicle_edge_transmission_power[vehicle_index][edge_index] = 0
                    else:
                        vehicle_edge_transmission_power[vehicle_index][edge_index] = vehicle_SINR[vehicle_index, -1] / (1 + vehicle_SINR[vehicle_index, -1]) / divided_sum * (edge_power - edge_occupied_power)
                else:
                    vehicle_edge_transmission_power[vehicle_index][edge_index] = vehicle_SINR[vehicle_index, -1] / (1 + vehicle_SINR[vehicle_index, -1]) / divided_sum * edge_power

            # for i in range(int(tasks_number_within_edge)):
            #     vehicle_index = vehicle_index_within_edge[i]
            #     if self._occuiped:
            #         if edge_power - edge_occupied_power <= 0:
            #             vehicle_edge_transmission_power[vehicle_index][edge_index] = 0
            #         else:
            #             vehicle_edge_transmission_power[vehicle_index][edge_index] = 1 / tasks_number_within_edge * (edge_power - edge_occupied_power)
            #     else:
            #         vehicle_edge_transmission_power[vehicle_index][edge_index] = 1 / tasks_number_within_edge * edge_power

        # print("vehicle_edge_transmission_power: ", vehicle_edge_transmission_power)
        return vehicle_edge_transmission_power
    
        
    # --------------------------- 奖励计算主流程 ---------------------------
    def compute_reward_with_convex_optimization(
        self,
        action: np.ndarray,
    ):
        """基于动作决定任务分配、功率与算力分配，并统计各类性能指标。"""
        # 将外部传入的动作转换为 numpy 数组，方便后续 reshape 与索引。
        actions = np.array(action)
        
        punished_time = 30
        # 若动作空间被展平，需要恢复为 (边缘节点 × 动作数) 的结构。
        if self._flatten_space:
            actions = np.reshape(np.array(actions), newshape=(self._config.edge_number, self._config.action_size))

        vehicle_SINR = np.zeros((self._config.vehicle_number, self._config.edge_number + 1))
        vehicle_transmission_time = np.zeros((self._config.vehicle_number, self._config.edge_number + 1))
        vehicle_execution_time = np.zeros((self._config.vehicle_number, self._config.edge_number + 1))
        vehicle_wired_transmission_time = np.zeros((self._config.vehicle_number, self._config.edge_number + 1))
        
        vehicle_intar_edge_inference = np.zeros((self._config.vehicle_number, self._config.edge_number + 1))
        vehicle_inter_edge_inference = np.zeros((self._config.vehicle_number, self._config.edge_number + 1))
        
        vehicle_edge_transmission_power = np.zeros((self._config.vehicle_number, self._config.edge_number))
        vehicle_edge_task_assignment = np.zeros((self._config.vehicle_number, self._config.edge_number))
        vehicle_edge_computation_resources = np.zeros((self._config.vehicle_number, self._config.edge_number))    
                
        # 统计指标用于构建环境返回的诊断信息，辅助分析算法表现。
        cumulative_reward = 0    
        successful_serviced_number = 0
        task_required_number = 0
        
        task_offloaded_number = 0
        
        # 先根据上一函数的结果得到更合理的发射功率分配，为计算 SINR 做准备。
        vehicle_edge_transmission_power = self.get_transmission_power_with_convex_optimization()
        
        for edge_index in range(self._config.edge_number):
            vehicle_index_within_edge = self._vehicle_index_within_edges[edge_index][self._time_slots.now()]
            vehicle_observed_index_within_edge = self._vehicle_observed_index_within_edges[edge_index][self._time_slots.now()]
            vehicle_number_within_edge = len(vehicle_observed_index_within_edge)
            
            the_edge = self._edge_list.get_edge_by_index(edge_index)

            # 针对每个边缘节点，提取其负责的车辆任务分配策略。
            task_assignment = np.array(actions[edge_index, :])
            task_assignment = np.reshape(task_assignment, newshape=(self._config.vehicle_number_within_edges, self._config.edge_number))
            
            for i in range(int(vehicle_number_within_edge)):
                processing_edge_index = int(task_assignment[i, :].argmax())
                
                vehicle_index = vehicle_observed_index_within_edge[i]                    
                # 仅对当前时间片真正发起任务的车辆进行调度。
                if self._vehicle_list.get_vehicle_by_index(vehicle_index).get_requested_task_by_slot_index(self._time_slots.now()) != -1:
                    
                    vehicle_edge_task_assignment[vehicle_index][processing_edge_index] = 1
                
                    if processing_edge_index != edge_index:
                        # 记录越区处理的任务数量，用于分析网络卸载情况。
                        task_offloaded_number += 1
                        task_index = self._vehicle_list.get_vehicle_by_index(vehicle_index).get_requested_task_by_slot_index(self._time_slots.now())
                        data_size = self._task_list.get_task_by_index(task_index).get_data_size()
                        # 计算跨边缘节点传输任务数据所需的有线时延。
                        wired_transmission_time = data_size / self._config.wired_transmission_rate * self._config.wired_transmission_discount * \
                                the_edge.get_edge_location().get_distance(self._edge_list.get_edge_by_index(processing_edge_index).get_edge_location())
                        for e in range(self._config.edge_number + 1):
                            vehicle_wired_transmission_time[vehicle_index, e] = wired_transmission_time
        
        for edge_index in range(self._config.edge_number):

            edge_computing_speed = self._edge_list.get_edge_by_index(edge_index).get_computing_speed()
            edge_occupied_computing_speed = self._occupied_computing_resources[edge_index][self._time_slots.now()]
            
            # 统计当前边缘节点需要处理的任务数量，以便平均分配算力。
            task_sum = int(np.sum(vehicle_edge_task_assignment[:, edge_index]))
            
            task_vehicle_index = np.where(vehicle_edge_task_assignment[:, edge_index] == 1)[0]
            
            # 先根据任务数据量和计算量构建权重，随后归一化得到算力分配比例。
            task_computation_resource_allocation = np.zeros((task_sum, ))
            for i in range(task_sum):
                vehicle_index = task_vehicle_index[i]
                task_index = self._vehicle_list.get_vehicle_by_index(vehicle_index).get_requested_task_by_slot_index(self._time_slots.now())
                data_size = self._task_list.get_task_by_index(task_index).get_data_size()
                computation_cycles = self._task_list.get_task_by_index(task_index).get_computation_cycles()
                task_computation_resource_allocation[i] = np.sqrt(edge_computing_speed * data_size * computation_cycles)
            new_task_computation_resource_allocation = (task_computation_resource_allocation) / np.sum((task_computation_resource_allocation))

            for i in range(task_sum):
                vehicle_index = task_vehicle_index[i]
                if self._occuiped:
                    if edge_computing_speed - edge_occupied_computing_speed <= 0:
                        vehicle_edge_computation_resources[vehicle_index][edge_index] = 0
                    else:
                        vehicle_edge_computation_resources[vehicle_index][edge_index] = new_task_computation_resource_allocation[i] * (edge_computing_speed - edge_occupied_computing_speed)
                else:
                    vehicle_edge_computation_resources[vehicle_index][edge_index] = new_task_computation_resource_allocation[i] * edge_computing_speed
                task_index = self._vehicle_list.get_vehicle_by_index(vehicle_index).get_requested_task_by_slot_index(self._time_slots.now())
                data_size = self._task_list.get_task_by_index(task_index).get_data_size()
                computation_cycles = self._task_list.get_task_by_index(task_index).get_computation_cycles()
                if vehicle_edge_computation_resources[vehicle_index][edge_index] != 0:
                    if float(data_size * computation_cycles / vehicle_edge_computation_resources[vehicle_index][edge_index]) < punished_time:
                        vehicle_execution_time[vehicle_index, -1] = float(data_size * computation_cycles / vehicle_edge_computation_resources[vehicle_index][edge_index])
                    else:
                        # 若分配算力不足导致时延过大，则按惩罚阈值截断。
                        vehicle_execution_time[vehicle_index, -1] = punished_time
                else:
                    vehicle_execution_time[vehicle_index, -1] = punished_time
                    
                for e in range(self._config.edge_number):  # e is the edge node which do nothing
                    if e == edge_index:
                        vehicle_execution_time[vehicle_index, e] = punished_time
                    else:
                        vehicle_execution_time[vehicle_index, e] = vehicle_execution_time[vehicle_index, -1]
                        
                # 在考虑资源占用的设置下，需要将后续时间片的资源提前扣除。
                if self._occuiped:
                    if vehicle_edge_computation_resources[vehicle_index][edge_index] != 0:
                        occupied_time = int(np.floor(data_size * computation_cycles / vehicle_edge_computation_resources[vehicle_index][edge_index]))
                        if self._occuiped and occupied_time > 0:
                            start_time = int(self._time_slots.now() + 1)
                            end_time = int(self._time_slots.now() + occupied_time + 1)
                            if end_time < self._config.time_slot_number:
                                for i in range(start_time, end_time):
                                    self._occupied_computing_resources[edge_index][i] += vehicle_edge_computation_resources[vehicle_index][edge_index]
                            else:
                                for i in range(start_time, int(self._config.time_slot_number)):
                                    self._occupied_computing_resources[edge_index][i] += vehicle_edge_computation_resources[vehicle_index][edge_index]
            
        """Compute the inference"""
        for edge_index in range(self._config.edge_number):
            
            vehicle_index_within_edge = self._vehicle_index_within_edges[edge_index][self._time_slots.now()]
            
            edge_inter_interference = np.zeros((self._config.edge_number))
            
            for other_edge_index in range(self._config.edge_number):
                if other_edge_index != edge_index:
                    vehicle_index_within_other_edge = self._vehicle_index_within_edges[other_edge_index][self._time_slots.now()]
                    for other_vehicle_index in vehicle_index_within_other_edge:
                        other_channel_condition = self._channel_condition_matrix[other_vehicle_index][edge_index][self._time_slots.now()]
                        inter_interference = np.power(np.absolute(other_channel_condition), 2) * cover_mW_to_W(vehicle_edge_transmission_power[other_vehicle_index][other_edge_index])
                        edge_inter_interference[other_edge_index] += inter_interference
            
            if vehicle_index_within_edge != []:
                for vehicle_index in vehicle_index_within_edge:
                    for e in range(self._config.edge_number):
                        vehicle_inter_edge_inference[vehicle_index, -1] += edge_inter_interference[e]
                        for other_edge_index in range(self._config.edge_number):
                            if e == other_edge_index:
                                vehicle_inter_edge_inference[vehicle_index, other_edge_index] += 0
                            else:
                                vehicle_inter_edge_inference[vehicle_index, other_edge_index] += edge_inter_interference[e]
                    channel_condition = self._channel_condition_matrix[vehicle_index][edge_index][self._time_slots.now()]
                    for other_vehicle_index in vehicle_index_within_edge:
                        if other_vehicle_index != vehicle_index:
                            other_channel_condition = self._channel_condition_matrix[other_vehicle_index][edge_index][self._time_slots.now()]
                            if other_channel_condition < channel_condition:
                                vehicle_intar_edge_inference[vehicle_index, -1] += np.power(np.absolute(other_channel_condition), 2) * cover_mW_to_W(vehicle_edge_transmission_power[other_vehicle_index][edge_index])
                    for e in range(self._config.edge_number):
                        if e == edge_index:
                            vehicle_intar_edge_inference[vehicle_index, e] = 0
                        else:
                            vehicle_intar_edge_inference[vehicle_index, e] = vehicle_intar_edge_inference[vehicle_index, -1]
                            
        """Compute the SINR and transimission time"""
        for edge_index in range(self._config.edge_number):
            if self._vehicle_index_within_edges[edge_index][self._time_slots.now()] != []:
                for vehicle_index in self._vehicle_index_within_edges[edge_index][self._time_slots.now()]:
                    task_index = self._vehicle_list.get_vehicle_by_index(vehicle_index).get_requested_task_by_slot_index(self._time_slots.now())
                    data_size = self._task_list.get_task_by_index(task_index).get_data_size()
                    
                    for e in range(self._config.edge_number):
                        if e == edge_index:
                        # 如果车辆尝试连接原所属边缘节点，则视为无效动作并施加惩罚时延。
                            vehicle_transmission_time[vehicle_index, e] = punished_time
                        else:
                            vehicle_SINR[vehicle_index, e] = compute_SINR(
                                white_gaussian_noise=self._config.white_gaussian_noise, 
                                channel_condition=self._channel_condition_matrix[vehicle_index][edge_index][self._time_slots.now()],
                                transmission_power=vehicle_edge_transmission_power[vehicle_index][edge_index],
                                intra_edge_interference=vehicle_intar_edge_inference[vehicle_index][e],
                                inter_edge_interference=vehicle_inter_edge_inference[vehicle_index][e],)
                            transmission_rate = compute_transmission_rate(
                                SINR=vehicle_SINR[vehicle_index, e], 
                                bandwidth=self._config.edge_bandwidth)
                            if transmission_rate != 0:
                                if float(data_size / transmission_rate) < punished_time:
                                    vehicle_transmission_time[vehicle_index, e] = float(data_size / transmission_rate)
                                else:
                                    vehicle_transmission_time[vehicle_index, e] = punished_time
                            else:
                                vehicle_transmission_time[vehicle_index, e] = punished_time
                    
                    vehicle_SINR[vehicle_index, -1] = compute_SINR(
                        white_gaussian_noise=self._config.white_gaussian_noise, 
                        channel_condition=self._channel_condition_matrix[vehicle_index][edge_index][self._time_slots.now()],
                        transmission_power=vehicle_edge_transmission_power[vehicle_index][edge_index],
                        intra_edge_interference=vehicle_intar_edge_inference[vehicle_index][-1],
                        inter_edge_interference=vehicle_inter_edge_inference[vehicle_index][-1],)
                    
                    transmission_rate = compute_transmission_rate(
                        SINR=vehicle_SINR[vehicle_index, -1], 
                        bandwidth=self._config.edge_bandwidth)
                    
                    if transmission_rate != 0:
                        if float(data_size / transmission_rate) < punished_time:
                            vehicle_transmission_time[vehicle_index, -1] = float(data_size / transmission_rate)
                        else:
                            vehicle_transmission_time[vehicle_index, -1] = punished_time
                    else:
                        vehicle_transmission_time[vehicle_index, -1] = punished_time
                    
                    # myapp.debug(f"data_size: {data_size}")
                    # myapp.debug(f"transmission_rate: {transmission_rate}")
                    # myapp.debug(f"transmission_time: {data_size / transmission_rate}")
                    # print("data_size: ", data_size)
                    # print("transmission_rate: ", transmission_rate)
                    # print("transmission_time: ", data_size / transmission_rate)
                    
                    if self._occuiped:
                        if transmission_rate != 0:
                            occupied_time = int(np.floor(data_size / transmission_rate))
                            if self._occuiped and occupied_time > 0:
                                start_time = int(self._time_slots.now() + 1)
                                end_time = int(self._time_slots.now() + occupied_time + 1)
                                if end_time < self._config.time_slot_number:
                                    for i in range(start_time, end_time):
                                        self._occupied_power[edge_index][i] += vehicle_edge_transmission_power[vehicle_index][edge_index]
                                else:
                                    for i in range(start_time, int(self._config.time_slot_number)):
                                        self._occupied_power[edge_index][i] += vehicle_edge_transmission_power[vehicle_index][edge_index]        

        successful_serviced = np.zeros(self._config.edge_number + 1)    
        
        service_time = np.zeros(self._config.edge_number + 1)        
        rewards = np.zeros(self._config.edge_number + 1)
        edge_task_requested_number = np.zeros(self._config.edge_number)
    
        average_transmision_time = 0
        average_wired_transmission_time = 0        
        average_execution_time = 0
        
        for edge_index in range(self._config.edge_number):
            if self._vehicle_index_within_edges[edge_index][self._time_slots.now()] != []:
                for vehicle_index in self._vehicle_index_within_edges[edge_index][self._time_slots.now()]:
                    task_index = self._vehicle_list.get_vehicle_by_index(vehicle_index).get_requested_task_by_slot_index(self._time_slots.now())
                    task_required_number += 1
                    task_service_time = vehicle_transmission_time[vehicle_index, -1] + vehicle_wired_transmission_time[vehicle_index, -1] + vehicle_execution_time[vehicle_index, -1]
                    average_transmision_time += vehicle_transmission_time[vehicle_index, -1]
                    average_wired_transmission_time += vehicle_wired_transmission_time[vehicle_index, -1]
                    average_execution_time += vehicle_execution_time[vehicle_index, -1]
                    service_time[-1] += task_service_time
                    if task_service_time <= self._task_list.get_task_by_index(task_index).get_delay_threshold():
                        successful_serviced_number += 1
                        successful_serviced[-1] += 1               
                    for e in range(self._config.edge_number):
                        if e != edge_index:
                            edge_task_requested_number[e] += 1
                            task_service_time = vehicle_transmission_time[vehicle_index, e] + vehicle_wired_transmission_time[vehicle_index, e] + vehicle_execution_time[vehicle_index, e]
                            service_time[e] += task_service_time
                            if task_service_time <= self._task_list.get_task_by_index(task_index).get_delay_threshold():
                                successful_serviced[e] += 1      

        rewards[-1] = successful_serviced[-1] / task_required_number
        for edge_index in range(self._config.edge_number):
            rewards[edge_index] = successful_serviced[edge_index] / task_required_number
        for edge_index in range(self._config.edge_number):
            rewards[edge_index] = rewards[-1] - rewards[edge_index]
    
        
        cumulative_reward = successful_serviced[-1] / task_required_number
        
        average_vehicle_SINR = np.sum(vehicle_SINR[:, -1])
        
        average_vehicle_intar_interference = np.sum(vehicle_intar_edge_inference[:, -1])
        average_vehicle_inter_interference = np.sum(vehicle_inter_edge_inference[:, -1])
        average_vehicle_interference = average_vehicle_intar_interference + average_vehicle_inter_interference
                
        average_service_time = average_transmision_time + average_wired_transmission_time + average_execution_time
        
        # print("successful_serviced_number: ", successful_serviced_number)
        # print("task_required_number: ", task_required_number)
        
        # 返回值包含每个边缘节点的即时奖励以及一系列诊断性统计指标。
        return rewards, cumulative_reward, average_vehicle_SINR, average_vehicle_intar_interference, average_vehicle_inter_interference, average_vehicle_interference, average_transmision_time, average_wired_transmission_time, average_execution_time, average_service_time, successful_serviced_number, task_offloaded_number, task_required_number
    

    """Define the action spaces of edge in critic network."""
    # --------------------- 规格定义（action/observation） ---------------------
    def critic_network_action_spec(self) -> specs.BoundedArray:
        # 为集中式 critic 网络提供展平的联合动作空间。
        """Define and return the action space."""
        critic_network_action_shape = (self._config.critic_network_action_size, )
        return specs.BoundedArray(
            shape=(critic_network_action_shape),
            dtype=float,
            minimum=np.zeros(critic_network_action_shape),
            maximum=np.ones(critic_network_action_shape),
            name='critic_actions',
        )

    """Define the gloabl observation spaces."""
    def observation_spec(self) -> specs.BoundedArray:
        # 定义多边缘节点联合观测的取值范围。根据是否考虑资源占用调整特征维度。
        """Define and return the observation space."""
        if self._occuiped:
            observation_size = self._config.observation_size
            if not self._for_mad5pg:
                observation_size -= 2
            observation_shape = (self._config.edge_number, observation_size)
            if self._flatten_space:
                observation_shape = (self._config.edge_number * observation_size, ) 
        else:
            observation_size = self._config.observation_size - 2 * self._config.edge_number
            if not self._for_mad5pg:
                observation_size -= 2
            observation_shape = (self._config.edge_number, observation_size)
            if self._flatten_space:
                observation_shape = (self._config.edge_number * observation_size, )
        return specs.BoundedArray(
            shape=observation_shape,
            dtype=float,
            minimum=np.zeros(observation_shape),
            maximum=np.ones(observation_shape),
            name='observations'
        )
    
    def edge_observation_spec(self) -> specs.BoundedArray:
        # 单个边缘节点视角的观测，用于分布式训练。
        """Define and return the observation space."""
        if self._occuiped:
            observation_size = self._config.observation_size
            if not self._for_mad5pg:
                observation_size -= 2
        else:
            observation_size = self._config.observation_size - 2 * self._config.edge_number
            if not self._for_mad5pg:
                observation_size -= 2
        observation_shape = (observation_size, )
        return specs.BoundedArray(
            shape=observation_shape,
            dtype=float,
            minimum=np.zeros(observation_shape),
            maximum=np.ones(observation_shape),
            name='edge_observations'
        )
    
    """Define the gloabl action spaces."""
    def action_spec(self) -> specs.BoundedArray:
        # 智能体输出的任务分配策略，范围在 [0,1] 之间。
        """Define and return the action space."""
        action_shape = (self._config.edge_number, self._config.action_size)
        if self._flatten_space:
            action_shape = (self._config.edge_number * self._config.action_size, )
        return specs.BoundedArray(
            shape=action_shape,
            dtype=float,
            minimum=np.zeros(action_shape),
            maximum=np.ones(action_shape),
            name='actions'
        )

    def edge_action_spec(self) -> specs.BoundedArray:
        # 针对单边缘节点的动作空间描述。
        """Define and return the action space."""
        action_shape = (self._config.action_size, )
        return specs.BoundedArray(
            shape=action_shape,
            dtype=float,
            minimum=np.zeros(action_shape),
            maximum=np.ones(action_shape),
            name='actions'
        )
    
    def reward_spec(self):
        # 奖励空间包含每个边缘节点局部奖励及全局奖励。
        """Define and return the reward space."""
        reward_shape = (self._config.reward_size, )
        return specs.Array(
            shape=reward_shape, 
            dtype=float, 
            name='rewards'
        )
    
    # ----------------------------- 观测构造逻辑 -----------------------------
    def _observation(self) -> np.ndarray:
        # 逐边缘节点拼接车辆信息、任务需求以及资源状态，构成强化学习的输入特征。
        """Return the observation of the environment."""
        """
        observation_shape = (self._config.edge_number, self._config.observation_size)
        The observation space is the location, task size, and computation cycles of each vehicle, then the aviliable transmission power, and computation resoucers
        """
        if self._occuiped:
            observation_size = self._config.observation_size
            if not self._for_mad5pg:
                observation_size -= 2
        else:
            observation_size = self._config.observation_size - 2 * self._config.edge_number
            if not self._for_mad5pg:
                observation_size -= 2
        # print("self._config.observation_size: ", self._config.observation_size)
        # print("self._for_mad5pg: ", self._for_mad5pg)
        # print("observation_size: ", observation_size)
        # 观测矩阵的维度随配置变化，可根据需要展平成一维数组。
        observation = np.zeros(shape=(self._config.edge_number, observation_size))
        
        for j in range(self._config.edge_number):
            vehicle_observed_index_within_edges = self._vehicle_observed_index_within_edges[j][self._time_slots.now()]
            vehicle_index_within_edges = self._vehicle_index_within_edges[j][self._time_slots.now()]
            vehicle_number_in_edge = len(vehicle_observed_index_within_edges)
            index = 0
            # if vehicle_number_in_edge != 3:
            #     print("vehicle_number_in_edge: ", len(vehicle_index_within_edges))
            for i in range(vehicle_number_in_edge):
                try:
                    # 获取车辆索引及其与当前边缘节点的距离。
                    vehicle_index = vehicle_observed_index_within_edges[i]
                    distance = self._distance_matrix[vehicle_index][j][self._time_slots.now()]
                    task_index = self._vehicle_list.get_vehicle_by_index(vehicle_index=vehicle_index).get_requested_task_by_slot_index(slot_index=self._time_slots.now())
                    data_size = self._task_list.get_task_by_index(task_index=task_index).get_data_size()
                    computing_cycles = self._task_list.get_task_by_index(task_index=task_index).get_computation_cycles()
                    delay_threshold = self._task_list.get_task_by_index(task_index=task_index).get_delay_threshold()
                    observation[j][index] = float(vehicle_index / self._config.vehicle_number)
                    index += 1
                    observation[j][index] = float(distance / self._config.communication_range)
                    index += 1
                    # -1 表示该车辆当前未请求任务，填充占位特征。
                    if task_index == -1:
                        observation[j][index] = 0
                        index += 1
                        observation[j][index] = 0
                        index += 1
                        # observation[j][index] = 0
                        # index += 1
                        observation[j][index] = 0
                        index += 1
                    else:
                        observation[j][index] = 1
                        index += 1
                        observation[j][index] = float((data_size - self._config.task_minimum_data_size) / (self._config.task_maximum_data_size - self._config.task_minimum_data_size))
                        index += 1
                        # observation[j][index] = float((computing_cycles  - self._config.task_minimum_computation_cycles) / (self._config.task_maximum_computation_cycles - self._config.task_minimum_computation_cycles))
                        # index += 1
                        observation[j][index] = float((delay_threshold - self._config.task_minimum_delay_thresholds) / (self._config.task_maximum_delay_thresholds - self._config.task_minimum_delay_thresholds))
                        index += 1
                except IndexError:
                    pass
            # print("index 1: ", index)
            index = self._config.vehicle_number_within_edges * 5
            # 拼接所有边缘节点的可用算力，帮助智能体做全局资源评估。
            for i in range(self._config.edge_number):
                edge_compuation_speed = self._edge_list.get_edge_by_index(edge_index=i).get_computing_speed()
                observation[j][index] = float((edge_compuation_speed - self._config.edge_minimum_computing_cycles)/ (self._config.edge_maximum_computing_cycles - self._config.edge_minimum_computing_cycles))
                index += 1
            # print("index 2: ", index)
            # 若考虑资源占用，还需提供当前时间片中各边缘节点的剩余功率和算力。
            if self._occuiped and not self._for_mad5pg:
                for i in range(self._config.edge_number):
                    observation[j][index] = self._occupied_power[i][self._time_slots.now()] / ( 1.01 * self._config.edge_power)
                    index += 1
                    observation[j][index] = self._occupied_computing_resources[i][self._time_slots.now()] / ( 1.01 * self._config.edge_maximum_computing_cycles)
                    index += 1
            if self._for_mad5pg and not self._occuiped:
                observation[j][-2] = self._time_slots.now() / (self._config.time_slot_number - 1)
                observation[j][-1] = j / (self._config.edge_number - 1)
            # print("index 3: ", index)
            if self._occuiped and self._for_mad5pg:
                for i in range(self._config.edge_number):
                    observation[j][index] = self._occupied_power[i][self._time_slots.now()] / ( 1.01 * self._config.edge_power)
                    index += 1
                    observation[j][index] = self._occupied_computing_resources[i][self._time_slots.now()] / ( 1.01 * self._config.edge_maximum_computing_cycles)
                    index += 1
                observation[j][-2] = self._time_slots.now() / (self._config.time_slot_number - 1)
                observation[j][-1] = j / (self._config.edge_number - 1)
        # 根据算法需求，可在此处将二维观测矩阵转换为一维。
        if self._flatten_space:
            if self._occuiped:
                observation_size = self._config.observation_size
                if not self._for_mad5pg:
                    observation_size -= 2
                    observation = np.reshape(observation, newshape=(self._config.edge_number * observation_size, ))
                else:
                    observation = np.reshape(observation, newshape=(self._config.edge_number * observation_size, ))
            else:
                observation_size = self._config.observation_size - 2 * self._config.edge_number
                if not self._for_mad5pg:
                    observation_size -= 2
                    observation = np.reshape(observation, newshape=(self._config.edge_number * observation_size, ))
                else:
                    observation = np.reshape(observation, newshape=(self._config.edge_number * observation_size, ))
        
        # print("observation: ")
        # print(observation)
        
        return observation


# --------------------------- 环境规格封装 ---------------------------
Array = specs.Array
BoundedArray = specs.BoundedArray
DiscreteArray = specs.DiscreteArray


# 将观测、动作、奖励等规格打包成 NamedTuple，便于上层框架统一处理。
class EnvironmentSpec(NamedTuple):
    """Full specification of the domains used by a given environment."""
    observations: NestedSpec
    edge_observations: NestedSpec
    critic_actions: NestedSpec
    actions: NestedSpec
    edge_actions: NestedSpec
    rewards: NestedSpec
    discounts: NestedSpec


# 该辅助函数生成 EnvironmentSpec，供外部快速了解环境的观测/动作/奖励结构。
def make_environment_spec(environment: vehicularNetworkEnv) -> EnvironmentSpec:
    """将环境提供的各类规格统一封装，便于算法初始化。"""
    """Returns an `EnvironmentSpec` describing values used by an environment."""
    return EnvironmentSpec(
        observations=environment.observation_spec(),
        edge_observations=environment.edge_observation_spec(),
        critic_actions=environment.critic_network_action_spec(),
        actions=environment.action_spec(),
        edge_actions=environment.edge_action_spec(),
        rewards=environment.reward_spec(),
        discounts=environment.discount_spec())
    

# 根据车辆与边缘节点数量推导动作、观测、奖励的维度，用于初始化配置。
def define_size_of_spaces(
    vehicle_number_within_edges: int,
    edge_number: int,
    task_assigned_number: Optional[int] = None,
) -> Tuple[int, int, int, int]:
    """The action space is task assignment"""
    # action_size for mad4pg
    # action_size 对应每辆车选择服务边缘节点的概率分布。
    action_size = vehicle_number_within_edges * edge_number
    
    # action_size = vehicle_number_within_edges + vehicle_number_within_edges * edge_number
    
    """The observation space is the location, task size, computing cycles of each vehicle, then the aviliable transmission power, and computation resoucers"""
    # observation_size 由车辆状态(5个特征/车)与边缘资源信息组成。
    observation_size = vehicle_number_within_edges * 5 + edge_number * 3 + 2
    
    """The reward space is the reward of each edge node and the gloabl reward
    reward[-1] is the global reward.
    reward[0:edge_number] are the edge rewards.
    """
    # reward_size 包含每个边缘节点局部奖励以及全局奖励。
    reward_size = edge_number + 1
    
    """Defined the shape of the action space in critic network"""
    # critic 网络需要观察所有边缘节点联合动作，因此尺寸为 edge_number * action_size。
    critici_network_action_size = edge_number * action_size
    
    return action_size, observation_size, reward_size, critici_network_action_size

    
# 预先计算车辆与边缘节点的距离及无线覆盖关系，减少仿真时的重复计算。
def init_distance_matrix_and_radio_coverage_matrix(
    env_config: env_config,
    vehicle_list: vehicleList,
    edge_list: edgeList,
) -> Tuple[np.ndarray, np.ndarray, List[List[List[int]]]]:
    """生成距离矩阵、信道条件矩阵以及每个时间片可服务车辆的索引列表。"""
    """Initialize the distance matrix and radio coverage."""
    # 距离矩阵的维度为 (车辆数, 边缘节点数, 时间片数)，用于快速查询。
    matrix_shpae = (env_config.vehicle_number, env_config.edge_number, env_config.time_slot_number)
    distance_matrix = np.zeros(matrix_shpae)
    channel_condition_matrix = [[[[] for _ in range(env_config.time_slot_number)] for _ in range(env_config.edge_number)] for _ in range(env_config.vehicle_number)]
    """Get the radio coverage information of each edge node."""
    # vehicle_index_within_edges 记录每个时间片内可被某边缘节点实际服务的车辆索引。
    vehicle_index_within_edges = [[[] for __ in range(env_config.time_slot_number)] for _ in range(env_config.edge_number)]
    vehicle_observed_index_within_edges = [[[] for __ in range(env_config.time_slot_number)] for _ in range(env_config.edge_number)]
    for i in range(env_config.vehicle_number):
        for j in range(env_config.edge_number):
            for k in range(env_config.time_slot_number):
                # 通过车辆轨迹与边缘节点位置计算欧氏距离。
                distance = vehicle_list.get_vehicle_by_index(i).get_distance_between_edge(k, edge_list.get_edge_by_index(j).get_edge_location())
                distance_matrix[i][j][k] = distance
                # 利用瑞利衰落与路径损耗模型生成复数信道增益。
                channel_condition_matrix[i][j][k] = compute_channel_gain(
                    rayleigh_distributed_small_scale_fading=generate_complex_normal_distribution(),
                    distance=distance,
                    path_loss_exponent=env_config.path_loss_exponent,
                )
                # 距离在覆盖半径内的车辆才会被加入候选列表。
                if distance_matrix[i][j][k] <= env_config.communication_range:
                    requested_task_index = vehicle_list.get_vehicle_by_index(i).get_requested_task_by_slot_index(k)
                    # 记录当前时间片可被观测到的车辆索引，任务是否存在由下方判断。
                    vehicle_observed_index_within_edges[j][k].append(i)
                    if requested_task_index != -1:
                        vehicle_index_within_edges[j][k].append(i)
    return distance_matrix, channel_condition_matrix, vehicle_index_within_edges, vehicle_observed_index_within_edges
    
