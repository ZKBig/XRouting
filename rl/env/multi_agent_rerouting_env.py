# -*- coding: utf-8 -*-
# @Description: dynamic rerouting environment used to test multi-agent
# @author: victor
# @create time: 2022-07-27-21:25
import numpy as np
import sys
from queue import PriorityQueue
import xlrd

from gym.spaces.box import Box
from gym.spaces.dict import Dict
from gym.spaces.discrete import Discrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv

sys.path.append('/usr/local/Cellar/sumo/1.10.0/share/sumo/tools')

import traci
import sumolib


class MultiAgentReroutingEnv(MultiAgentEnv):
    """
    This class is used to define the dynamic routing environment
    """

    def __init__(self,
                 observation_size,
                 action_size,
                 work_dir,
                 destination,
                 initial_edge,
                 model="XRouting",
                 nogui=False,
                 ):

        """
        Initialize the multi-agent dynamic rerouting environment class

        :param observation_size: The size of the observation space
        :param action_size: The size of the action space
        :param work_dir: the absolute directory of work place
        :param nogui: control whether displaying the GUI
        """

        self.observation_size = observation_size
        self.action_size = action_size
        self.nogui = nogui
        self.work_dir = work_dir
        self.destination = destination
        self.initial_edge = initial_edge
        self.route_id = "rl_route"
        self.model = model
        self.rl_car_num = 75
        self.agent_ids = set(["rl_{0}".format(i) for i in range(self.rl_car_num)])
        self.agents = ["rl_{0}".format(i) for i in range(self.rl_car_num)]
        self.dones = dict()
        self.rl_car_next_edge = dict()
        self.rl_car_current_edge = dict()
        self.sorted_edges = dict()

        # initialize dones list
        for i in range(len(self.agent_ids)):
            self.dones[self.agent_ids.pop()] = False

        # number to record the total simulation time
        self.time_interval_counter = 0

        # record the episode number
        self.episode_counter = 0

        self.initial_reset = 0

        # determine whether using sumo in GUI mode or not
        if self.nogui:
            self.sumoBinary = sumolib.checkBinary('sumo')
        else:
            self.sumoBinary = sumolib.checkBinary('sumo-gui')

        self.coordinates_dir = self.work_dir + "/sumo_scenario/edge_coordinates.xlsx"
        self.configuration_file_path = self.work_dir + "/sumo_scenario/scenario_rl.sumocfg.xml"
        self.net_dir = self.work_dir + "/sumo_scenario/scenario.net.xml"
        self.trip_info_dir = self.work_dir + "/training_tripinfo"
        self.tracefile_dir = self.work_dir + "/sumo_scenario/scenario_rl_log.txt"
        self.trip_dir = self.work_dir + "/sumo_scenario/scenario_rl.trips.xml"

        self.nextEdges = self._find_adjacent_edge(self.net_dir)
        self.destinations = self._find_final_edge(self.trip_dir)
        self.initial_edges = self._find_initial_edge(self.trip_dir)

        # read the coordinates of the end and the start of all edges
        self._read_edge_coordinates(self.coordinates_dir)

    def _read_edge_coordinates(self, coordinates_dir):
        """
        Read the coordinate of all values from the excel file
        :param coordinates_dir: the absolute directory of the excel file
        :return: Null
        """
        book = xlrd.open_workbook(coordinates_dir)
        sheet1 = book.sheets()[1]

        edge_ids = sheet1.col_values(0)
        start_edges = sheet1.col_values(1)
        end_edges = sheet1.col_values(3)

        self.edges_start_position = dict()
        self.edges_end_position = dict()

        for i in range(1, len(edge_ids)):
            self.edges_start_position[edge_ids[i]] = [float(start_edges[i].split(',')[0]),
                                                      float(start_edges[i].split(',')[1])]
            self.edges_end_position[edge_ids[i]] = [float(end_edges[i].split(',')[0]),
                                                    float(end_edges[i].split(',')[1])]

        # record the coordinates of the middle of the edge
        self.middle_point_coordinates = dict()

        for k, v in self.edges_start_position.items():
            x = (v[0] + self.edges_end_position[k][0]) / 2
            y = (v[1] + self.edges_end_position[k][1]) / 2
            self.middle_point_coordinates[k] = [x, y]

    def get_agent_ids(self):
        """
        Return a set of agent ids in the environment.

        Returns: Set of agent ids.
        """
        print(self.agent_ids)
        return self.agent_ids

    @property
    def observation_space(self):
        """
        define the observation apace in the rl
        At each step, we emit a dict of:
            - the actual cart observation
            - a mask of valid actions (e.g., [0, 0, 1, 0] for four max avail)
        :return: the observation space
        """
        if self.model == "XRouting":
            return Dict({
                "real_observation": Box(low=-1, high=1, shape=(self.observation_size, 6),
                                        dtype=np.float32),
                "action_mask": Box(0, 1, shape=(self.action_size,), dtype=np.float32),
                "position": Box(low=-1, high=1, shape=(self.observation_size, 1),
                                dtype=np.float32)
            })
        else:
            return Dict({
                "real_observation": Box(low=-1, high=1, shape=(self.observation_size * 2 + 4,),
                                        dtype=np.float32),
                "action_mask": Box(0, 1, shape=(self.action_size,), dtype=np.float32),
            })

    @property
    def action_space(self):
        """
        define the action spcae in the rl
        :return: the action space
        """
        return Discrete(4)

    def reset(self):
        """
        reset the sumo simulation every episode
        :return: null
        """

        print("****************************RESET*****************************")
        print("**************************************************************")
        print("**************************************************************")

        # define the directory of the tripinfo files
        if self.model == "XRouting":
            trip_info_file = self.trip_info_dir + "/XRouting_training/tripinfo_eval.xml"
        elif self.model == "PPO":
            trip_info_file = self.trip_info_dir + "/PPO_training/tripinfo_eval.xml"
        else:
            trip_info_file = self.trip_info_dir + "/DQN_training/tripinfo_eval.xml"

        # record the number of the episodes
        self.episode_counter += 1

        if self.initial_reset == 0:
            traci.start([self.sumoBinary, "-c",
                         self.configuration_file_path,
                         "--tripinfo-output.write-unfinished", "True",
                         "--tripinfo-output", trip_info_file, "--ignore-route-errors"],
                        traceFile=self.tracefile_dir)

        self.initial_reset += 1

        # initialize actions of all rl vehicles to 1
        actions = dict()
        for i in range(len(self.agent_ids)):
            actions["rl_{0}".format(i)] = 1

        obs, _, _, _ = self.step(rl_actions=actions)

        return obs

    def step(self, rl_actions):
        """
        Advance the environment by one step.
        :param rl_actions: the direction indices of all rl vehicles, that is, 0, 1, 2, 3, according to the current action-mask
        :return: obs, reward, done, {}
        """

        print("***************", rl_actions, "**************")
        rl_ids = []
        rl_action_list = []
        no_car = False

        # obtain the rl vehicles' ids and the corresponding actions for the further use
        for veh_id, action in rl_actions.items():
            rl_ids.append(veh_id)
            rl_action_list.append(action)

        vehicle_ids = traci.vehicle.getIDList()

        # update the next edge for each vehicle
        for i in range(len(rl_ids)):
            if rl_ids[i] in vehicle_ids:
                self._determine_next_edge(rl_ids[i], rl_action_list[i])

        if traci.simulation.getMinExpectedNumber() <= 0:
            no_car = True
            for i in range(len(rl_ids)):
                self.dones[rl_ids[i]] = True
        else:
            # update the action of each rl vehicle every 20 time-steps
            while self.time_interval_counter % 30 != 0 and \
                    traci.simulation.getMinExpectedNumber() > 0:
                traci.simulationStep()

                vehicle_ids_3 = traci.vehicle.getIDList()

                self.time_interval_counter += 1
                print("Inside", self.time_interval_counter)

                for i in range(len(rl_ids)):
                    if rl_ids[i] in vehicle_ids_3:
                        current_edge = traci.vehicle.getRoadID(rl_ids[i])
                        # print(rl_ids[i], '******************************:', current_edge)

        current_edges = dict()

        self.time_interval_counter += 1
        print("Outside:", self.time_interval_counter)

        # set done to True if the current episode is over or the collision occurs
        if traci.simulation.getMinExpectedNumber() <= 0 or \
                traci.simulation.getCollisions() is None:

            for i in range(len(rl_ids)):
                self.dones[rl_ids[i]] = True

        else:
            vehicle_ids_2 = traci.vehicle.getIDList()
            print(vehicle_ids_2)
            # get the current roads' id of all the rl vehicles
            for i in range(len(rl_ids)):
                if not self.dones[rl_ids[i]]:
                    if rl_ids[i] in vehicle_ids_2:
                        if traci.vehicle.getRoadID(rl_ids[i]) == self.destinations[rl_ids[i]]:
                            self.dones[rl_ids[i]] = True
                        else:
                            current_edges[rl_ids[i]] = traci.vehicle.getRoadID(rl_ids[i])
                            self.dones[rl_ids[i]] = False
                    elif (rl_ids[i] not in vehicle_ids_2) and (rl_ids[i] in vehicle_ids):
                        current_edges[rl_ids[i]] = self.destinations[rl_ids[i]]
                        self.dones[rl_ids[i]] = True
                    else:
                        current_edges[rl_ids[i]] = self.initial_edges[rl_ids[i]]
                        self.dones[rl_ids[i]] = False

        # update the action masks for all the rl vehicles
        action_masking = self._update_avail_actions(current_edges)

        states = dict()
        positions = dict()
        rewards = dict()
        for i in range(len(rl_ids)):
            if self.model == "XRouting":
                states[rl_ids[i]], positions[rl_ids[i]], self.sorted_edges[rl_ids[i]] = self._get_state(rl_ids[i])
            else:
                states[rl_ids[i]] = self._get_state(rl_ids[i])

            rewards[rl_ids[i]] = self.compute_reward()

        self.dones["__all__"] = True
        obs = {}

        if self.model == "XRouting":
            for i in range(len(rl_ids)):
                if self.dones[rl_ids[i]]:
                    observation = {"real_observation": [],
                                   "position": [],
                                   "action_mask": []}
                    obs.update({rl_ids[i]: observation})
                else:
                    observation = {"real_observation": states[rl_ids[i]],
                                   "position": positions[rl_ids[i]],
                                   "action_mask": action_masking[rl_ids[i]]}
                    obs.update({rl_ids[i]: observation})

            for i in range(len(self.agents)):
                if self.agents[i] not in rl_ids:
                    observation = {"real_observation": [],
                                   "position": [],
                                   "action_mask": []}
                    obs.update({self.agents[i]: observation})

            return obs, rewards, self.dones, {"no_car": no_car, "rl_edges": self.rl_car_next_edge,
                                              "rl_current_edges": self.rl_car_current_edge,
                                              "sorted_edges": self.sorted_edges}
        else:
            for i in range(len(rl_ids)):
                if self.dones[rl_ids[i]]:
                    observation = {"real_observation": [],
                                   "action_mask": []}
                    obs.update({rl_ids[i]: observation})
                else:
                    observation = {"real_observation": states[rl_ids[i]],
                                   "action_mask": action_masking[rl_ids[i]]}
                    obs.update({rl_ids[i]: observation})

            for i in range(len(self.agents)):
                if self.agents[i] not in rl_ids:
                    observation = {"real_observation": [],
                                   "action_mask": []}
                    obs.update({self.agents[i]: observation})

            return obs, rewards, self.dones, {"no_car": no_car}

    def _update_avail_actions(self, current_edges) -> dict:
        """
        update all the vehicles action mask
        """
        action_masking = dict()
        mapping = {"s": 0, "r": 1, "l": 2, "t": 3}

        for k, v in current_edges.items():
            action_mask = [0 for i in range(self.action_size)]

            next_edges = self.nextEdges[v]

            directions = list(next_edges.keys())

            for direction in mapping.keys():
                if direction in directions:
                    action_mask[mapping[direction]] = 1

            action_masking[k] = action_mask

        return action_masking

    def _determine_next_edge(self, rl_id, index):
        """
        determine the next edge of each vehicle
        """
        choices = {0: "r", 1: "s", 2: "l", 3: "t"}
        edge = traci.vehicle.getRoadID(rl_id)

        print("***********", rl_id, "current edge is:", edge, "***********")
        if edge == self.destinations[rl_id]:
            new_route = [edge]
            next_edge = self.destinations[rl_id]
        elif (edge == "A1A2" or edge == "B2A2") and self.destinations[rl_id] == self.destination:
            new_route = [edge, self.destination]
            next_edge = self.destination
        else:
            direction = choices[index]

            if direction in list(self.nextEdges[edge].keys()):
                next_edge = self.nextEdges[edge][direction]
            else:
                next_edge = self.nextEdges[edge][list(self.nextEdges[edge].keys())[0]]

            new_route = [edge, next_edge]

        self.rl_car_next_edge[rl_id] = next_edge
        self.rl_car_current_edge[rl_id] = edge

        traci.vehicle.setRoute(rl_id, new_route)

        return next_edge

    def _get_state(self, rl_id):
        """
        obtain the average speeds and vehicle density of each edge in the network
        :return: the state numpy array
        """

        # get all the edges in the network
        edges = traci.edge.getIDList()
        edges = [edge_id for edge_id in edges if ":" not in edge_id]

        # get all the vehicles in the network
        vehicles = traci.vehicle.getIDList()

        # container which records the average speed of each edge and is initialed as 0
        average_speed = [0 for i in range(len(edges))]
        # container which records the number of vehicles of each edge and is initialed as 0
        number = [0 for i in range(len(edges))]

        edge_length = dict()
        avr_speed_each_edge = dict(zip(edges, average_speed))
        num_vehicle_each_edge = dict(zip(edges, number))
        density_each_edge = dict(zip(edges, number))
        travel_time = dict(zip(edges, number))

        rl_vehicle = " "

        # sum the values of speed and the numbers of all the vehicles on each edge
        for veh_id in vehicles:
            edge = traci.vehicle.getRoadID(veh_id)
            if veh_id[0:2] == "rl":
                rl_vehicle = veh_id
            if edge[0] == ':':
                pass
            else:
                speed = traci.vehicle.getSpeed(veh_id)
                avr_speed_each_edge[edge] += speed
                num_vehicle_each_edge[edge] += 1

        # find the average speed and vehicle density of each edge
        for i in range(len(edges)):
            edge_id = edges[i]
            lane_id = edge_id + "_0"
            length = traci.lane.getLength(lane_id)
            edge_length[edge_id] = length
            if num_vehicle_each_edge[edge_id] == 0:
                avr_speed_each_edge[edge_id] = 0
            else:
                avr_speed_each_edge[edge_id] /= num_vehicle_each_edge[edge_id]
            density_each_edge[edge_id] = num_vehicle_each_edge[edge_id] / length
            travel_time[edge_id] = traci.edge.getTraveltime(edge_id)

        # list stores attribute v_{avr}
        edge_average_speed = []
        # list stores attribute d_{veh}
        edge_density = []
        # list stores attribute t_{avr}
        edge_travel_time = []
        # list stores attribute dis_{aim}
        edge_end_distance = []
        # list stores attribute position encoding P
        position_encoding = []
        # list stores 0/1 (destination: 1  else: 0)
        destination_mask = []
        # list stores attribute angle
        edge_angle = []

        destination_edge = self.destinations[rl_id]
        destination_cor = self.edges_end_position[destination_edge]

        # obtain the current position of the rl vehicle
        if rl_vehicle == " ":
            current_x = destination_cor[0]
            current_y = destination_cor[1]
        else:
            current_x, current_y = traci.vehicle.getPosition(rl_vehicle)

        # obtain the position encoding of the edges
        start_positions = dict()
        end_positions = dict()
        distance_approximation = dict()
        angle = dict()

        for k, v in self.edges_start_position.items():
            start_positions[k] = ((v[0] - current_x) ** 2 + (v[1] - current_y) ** 2) ** 0.5

        for k, v in self.edges_end_position.items():
            end_positions[k] = ((v[0] - 0.00) ** 2 + (v[1] - 454.47) ** 2) ** 0.5

        for k, v in edge_length.items():
            distance_approximation[k] = start_positions[k] + end_positions[k] + edge_length[k]

        # sort the attributes of all edges according to the distance between the rl vehicle and the destination
        sorted_positions = sorted(distance_approximation.items(), key=lambda x: x[1], reverse=False)

        # calculate the angle of each edge towards the rl vehicle
        for k, v in self.middle_point_coordinates.items():
            if v[0] - current_x == 0:
                angle[k] = np.pi / 2
            else:
                angle[k] = np.arctan((v[1] - current_y) / (v[0] - current_x))

        sorted_edge = []
        for item in sorted_positions:
            sorted_edge.append(item[0])
            position_encoding.append(item[1])
            edge_average_speed.append(avr_speed_each_edge[item[0]])
            edge_density.append(density_each_edge[item[0]])
            edge_travel_time.append(travel_time[item[0]])
            edge_angle.append(angle[item[0]])
            cor = self.edges_start_position[item[0]]
            distance = ((cor[0] - 0.00) ** 2 + (cor[1] - 454.47) ** 2) ** 0.5
            edge_end_distance.append(distance)
            if item[0] == destination_edge:
                destination_mask.append(1)
            else:
                destination_mask.append(0)

        # normalize all the input attributes
        if max(edge_average_speed) == 0:
            edge_average_speed_result = edge_average_speed
        else:
            edge_average_speed_result = [speed / max(edge_average_speed)
                                         for speed in edge_average_speed]

        if max(edge_density) == 0:
            edge_density_result = edge_density
        else:
            edge_density_result = [density / max(edge_density)
                                   for density in edge_density]

        if max(edge_end_distance) == 0:
            final_edge_end_distance = edge_end_distance
        else:
            final_edge_end_distance = [distance / max(edge_end_distance)
                                       for distance in edge_end_distance]

        if max(travel_time) == 0:
            edge_travel_time = travel_time
        else:
            edge_travel_time = [time / max(edge_travel_time)
                                for time in edge_travel_time]

        max_angle = 0
        for i in range(len(edge_angle)):
            if abs(edge_angle[i]) > max_angle:
                max_angle = abs(edge_angle[i])

        edge_angle = [angle / max_angle for angle in edge_angle]

        edge_state1 = list(zip(edge_average_speed_result, edge_density_result, final_edge_end_distance,
                               destination_mask, edge_angle, edge_travel_time))

        final_position_encoding = [[item / max(position_encoding)] for item in position_encoding]

        # ---------------------- PPO/DQN observation input ----------------------
        position = [current_x, current_y, destination_cor[0], destination_cor[1]]

        max_num = 0
        for i in range(len(position)):
            if abs(position[i]) > max_num:
                max_num = abs(position[i])

        pos = [axis / max_num for axis in position]
        edge_state2 = edge_average_speed_result + edge_density_result + pos
        # -----------------------------------------------------------------------

        if self.model == "XRouting":
            return np.array(edge_state1, dtype=np.float32), np.array(final_position_encoding, dtype=np.float32), \
                   sorted_edge
        else:
            return np.array(edge_state2, dtype=np.float32)

    def compute_reward(self):
        """
        Return the average delay for all vehicles in the system.
        Returns
        -------
        float
        reward value
        """
        return -1

    def _find_adjacent_edge(self, file_name):
        """
        obtain the adjacent edges of each edge, which are stored in a dictionary.
        for example:

        {'A0A1': {'r': 'A1B1', 's': 'A1A2', 'l': 'A1left1', 't': 'A1A0'},
        'A0B0': {'r': 'B0bottom1', 's': 'B0D5', 'l': 'B0B1', 't': 'B0A0'},
        'A0bottom0': {'t': 'bottom0A0'},
        'A0left0': {'t': 'left0A0'},
        'A1A0': {'r': 'A0left0', 's': 'A0bottom0', 'l': 'A0B0', 't': 'A0A1'}}

        'r', 's', 'l', 't' represents the direction of the adjacent edges.

        """
        attr_connection = {'connection': ['from', 'to', 'fromLane', 'toLane', 'via', 'tl', 'linkIndex', 'dir', 'state']}
        attr_edge = {'edge': ['id', 'from', 'to', 'priority']}

        # obtain all the edges in the network
        turn_info = dict()
        edges = list(sumolib.xml.parse(file_name, 'net', attr_edge))[0]
        for edge in edges.edge:
            next_edges = dict()
            turn_info[edge.id] = next_edges

        # obtain the adjacent edges of each edge
        connections = list(sumolib.xml.parse(file_name, 'net', attr_connection))[0]
        for connection in connections.connection:
            edge = connection.attr_from
            direction = connection.dir
            if direction == "L" or direction == "R":
                direction = "s"
            next_edge = connection.to
            turn_info[edge][direction] = next_edge

        return turn_info

    def _find_final_edge(self, file_name):
        trip_attr = {'trip': ['depart', 'from', 'to', 'id', 'type']}
        destination = dict()
        trips = list(sumolib.xml.parse(file_name, trip_attr))
        for trip in trips:
            destination[trip.id] = trip.to

        print("-------------The detination edge of each vehicle-------------")
        # print(destination)

        return destination

    def _find_initial_edge(self, file_name):
        trip_attr = {'trip': ['depart', 'from', 'to', 'id', 'type']}
        initial = dict()
        trips = list(sumolib.xml.parse(file_name, trip_attr))
        for trip in trips:
            initial[trip.id] = trip.attr_from
        print("-------------The initial edge of each vehicle-------------")
        # print(initial)

        return initial
