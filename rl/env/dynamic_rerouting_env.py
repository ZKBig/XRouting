# -*- coding: utf-8 -*-
# @Description: Traffic network environment class based on Gym
# @author: victor
# @create time: 2022-07-26-16:16

from __future__ import absolute_import
import numpy as np
import gym
import sys
import pandas as pd
import xlrd
import os

from gym.spaces.box import Box
from gym.spaces.dict import Dict
from gym.spaces.discrete import Discrete

sys.path.append('/usr/local/Cellar/sumo/1.10.0/share/sumo/tools')
import traci
import sumolib


class DynamicRerouteEnv(gym.Env):
    """
    This class is used to define the dynamic routing environment
    """

    def __init__(self,
                 observation_size,
                 action_size,
                 initial_edge,
                 destination,
                 work_dir,
                 model="XRouting",
                 nogui=True,
                 ):

        """
        Initialize the dynamic rerouting environment class

        :param observation_size: The size of the observation space
        :param action_size: The size of the action space
        :param initial_edge: The name of the initial edge
        :param destination: The name of the destination edge
        :param work_dir: The absolute direction of the work place
        :param nogui: control whether displaying the GUI
        """

        self.observation_size = observation_size
        self.action_size = action_size
        self.nogui = nogui
        self.route_id = "rl_route"
        self.initial_edge = initial_edge
        self.destination = destination
        self.model = model
        self.work_dir = work_dir

        # number to record the total simulation time
        self.time_counter = 0

        # number of total steps taken
        self.step_counter = 0

        # record the episode number
        self.episode_counter = 0

        # index number of rl vehicle
        self.rl_number = 0

        # reward recorder
        self.travel_time = 0

        self.reward = list()

        self.has_add = False

        self.initial_reset = 0

        # initial the id of the rl car to be removed
        self.rl_car_remove = " "

        # determine whether using sumo in GUI mode or not
        if self.nogui:
            self.sumoBinary = sumolib.checkBinary('sumo')
        else:
            self.sumoBinary = sumolib.checkBinary('sumo-gui')

        # read the coordinates of the end and the start of all edges

        self.coor_dir = self.work_dir + "/sumo_scenario/edge_coordinates.xlsx"
        self.configuration_file_path = self.work_dir + "/sumo_scenario/scenario.sumocfg.xml"
        self.net_dir = self.work_dir + "/sumo_scenario/scenario.net.xml"
        self.trip_info_dir = self.work_dir + "/training_tripinfo"
        self.tracefile_dir = self.work_dir + "/sumo_scenario/scenario_log.txt"

        self._read_edge_coordinates(self.coor_dir)
        self.nextEdges = self._find_adjacent_edge(self.net_dir)

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

        # update the action-mask
        self._update_avail_actions(self.initial_edge)

        print("****************************RESET*****************************")
        print("**************************************************************")
        print("**************************************************************")

        print(self.step_counter)
        self.step_counter = 0

        if self.time_counter != 0:
            traci.close(False)

        # define the directory of the tripinfo files
        if self.model == "XRouting":
            trip_info_file = self.trip_info_dir + "/XRouting_training/tripinfo" + str(self.episode_counter) + ".xml"
        elif self.model == "PPO":
            trip_info_file = self.trip_info_dir + "/PPO_training/tripinfo" + str(self.episode_counter) + ".xml"
        else:
            trip_info_file = self.trip_info_dir + "/DQN_training/tripinfo" + str(self.episode_counter) + ".xml"


        # record the number of the episodes
        self.episode_counter += 1

        traci.start([self.sumoBinary, "-c",
                     self.configuration_file_path,
                     "--tripinfo-output.write-unfinished", "True",
                     "--tripinfo-output", trip_info_file, "--ignore-route-errors"],
                    traceFile=self.tracefile_dir)

        self.has_add = False

        obs, _, _, _ = self.step(action=1)

        return obs

    def step(self, action):
        """
        Advance the environment by one step.
        :param action: the direction index, that is, 0, 1, 2, 3, according to the current action-mask
        :return: obs, reward, done, {}
        """

        self.step_counter += 1

        # mark whether the simulation is done
        done = False

        # judge whether the rl car is in decision zone
        detected = False

        # flag used to reset the value of travel_time to 1
        reset = False

        # flag used to remove the last rl vehicle when it reaches to the destination edge
        self.remove = False

        if traci.simulation.getMinExpectedNumber() <= 0:
            done = True
        else:
            while traci.simulation.getMinExpectedNumber() > 0 and not detected:
                # simulation advances one step
                traci.simulationStep()
                self.time_counter += 1
                print(self.time_counter)

                # remove the last rl vehicle
                if self.rl_car_remove != " ":
                    if self.remove and traci.vehicle.getRoadID(self.rl_car_remove) == self.destination:
                        traci.vehicle.remove(self.rl_car_remove, reason=2)
                        self.remove = False

                vehicles = traci.vehicle.getIDList()
                vehicle_type = [traci.vehicle.getTypeID(veh) for veh in vehicles]
                vehicle_type = list(set(vehicle_type))

                if self.has_add and "rl_car" in vehicle_type:
                    self.has_add = False

                rl_car_id = "rl" + str(self.rl_number)

                # if there is no rl car in the current network, then add a new rl car by assigning the default route
                if "rl_car" not in vehicle_type and not self.has_add and ("human_bus" in vehicle_type
                                                                          or "human_car" in vehicle_type):
                    traci.vehicle.add(vehID=rl_car_id, routeID=self.route_id,
                                      typeID="rl_car", depart='now',
                                      departLane='first', departPos='base',
                                      departSpeed='0', personCapacity=4,
                                      personNumber=3)
                    reset = True
                    self.has_add = True
                    self.rl_number += 1

                # detect the rl car after each simulation step
                detected, rl_id = self._rl_in_decision_zone()

                # accumulate the travel time until the rl car reaches to the decision zone
                if not reset:
                    self.travel_time += 1
                else:
                    self.travel_time = 0
                    reset = False

            if traci.simulation.getMinExpectedNumber() <= 0 or rl_id == ' ':
                done = True
            else:
                # determine the next edge of the rl vehicle
                next_edge = self._determine_next_edge(self.nextEdges, rl_id, action)

                if next_edge == self.destination or next_edge == " ":
                    self.remove = True
                    self.rl_car_remove = rl_id
                else:
                    # update the action mask
                    self._update_avail_actions(next_edge)

                # crash encodes whether the simulator experienced a collision
                crash = traci.simulation.getCollisions()

                # stop collecting new simulation steps if there is a collision
                if crash is None:
                    done = True

        # obtain the next observation state
        if self.model == "XRouting":
            state, position = self._get_state()
        else:
            state = self._get_state()

        # obtain the reward for this step
        reward = self.compute_reward()

        # recorde the reward of each action
        self.reward.append(reward)

        reward_out = pd.DataFrame(self.reward)

        if self.model == "XRouting":
            reward_out.to_csv(self.trip_info_dir + '/XRouting_training/reward_each_episode.csv', encoding='gbk')
        elif self.model == "PPO":
            reward_out.to_csv(self.trip_info_dir + '/PPO_training/reward_each_episode.csv', encoding='gbk')
        else:
            reward_out.to_csv(self.trip_info_dir + '/DQN_training/reward_each_episode.csv', encoding='gbk')

        # reset the travel time to 0
        self.travel_time = 0

        if self.model == "XRouting":
            obs = {
                "action_mask": self.action_mask,
                "real_observation": state,
                "position": position
            }
        else:
            obs = {
                "action_mask": self.action_mask,
                "real_observation": state,
            }

        return obs, reward, done, {}

    def _update_avail_actions(self, edge):
        """
        update the action-mask in each step according to the next edges dictionary
        """
        mapping = {"s": 0, "r": 1, "l": 2, "t": 3}
        self.action_mask = [0 for i in range(self.action_size)]

        next_edges = self.nextEdges[edge]

        directions = list(next_edges.keys())

        for direction in mapping.keys():
            if direction in directions:
                self.action_mask[mapping[direction]] = 1

        print(edge, "***********************************************************", self.action_mask)

    def _rl_in_decision_zone(self):
        """
        judge whether the rl car is in decision zone.
        return True and the rl car id if it is detected.
        """
        detected = False
        rl_id = " "

        detectors = traci.inductionloop.getIDList()
        for det in detectors:
            vehicle_IDs = traci.inductionloop.getLastStepVehicleIDs(det)
            for id in vehicle_IDs:
                if id[0:2] == "rl":
                    # if the rl car has already been in the destination edge,
                    # there is no need to process it
                    if traci.vehicle.getRoadID(id) == self.destination:
                        break
                    else:
                        detected = True
                        rl_id = id
                        break

        return detected, rl_id

    def _determine_next_edge(self, turn_info, id, index):
        """
        reroute all the vehicle detected by the sumo induction loop
        :param loop_name: the name of the induction
        :return:
        """

        print("***********action_index is:", index, "**************")

        choices = {0: "s", 1: "r", 2: "l", 3: "t"}
        edge = traci.vehicle.getRoadID(id)

        print(edge)
        if edge[0] == ':':
            next_edge = " "
        elif edge == self.destination:
            new_route = [edge]
            traci.vehicle.setRoute(id, new_route)
            next_edge = self.destination
        elif edge == "A2A3" or edge == "B3A3":
            new_route = [edge, self.destination]
            traci.vehicle.setRoute(id, new_route)
            next_edge = self.destination
        else:
            direction = choices[index]
            if direction in list(turn_info[edge].keys()):
                next_edge = turn_info[edge][direction]
            else:
                next_edge = turn_info[edge][list(turn_info[edge].keys())[0]]

            print("*********vehicle:", id, ", ", direction, ":", next_edge, "********")
            new_route = [edge, next_edge]
            traci.vehicle.setRoute(id, new_route)

        return next_edge

    def _get_state(self):
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

        # obtain the current position of the rl vehicle
        if rl_vehicle == " ":
            current_x, current_y = (0.00, 454.47)
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

        for item in sorted_positions:
            position_encoding.append(item[1])
            edge_average_speed.append(avr_speed_each_edge[item[0]])
            edge_density.append(density_each_edge[item[0]])
            edge_travel_time.append(travel_time[item[0]])
            edge_angle.append(angle[item[0]])
            cor = self.edges_start_position[item[0]]
            distance = ((cor[0] - 0.00) ** 2 + (cor[1] - 454.47) ** 2) ** 0.5
            edge_end_distance.append(distance)
            if item[0] == self.destination:
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
        position = [current_x, current_y, 0.00, 450.00]

        max_num = 0
        for i in range(len(position)):
            if abs(position[i]) > max_num:
                max_num = abs(position[i])

        pos = [axis / max_num for axis in position]
        edge_state2 = edge_average_speed_result + edge_density_result + pos
        # -----------------------------------------------------------------------

        if self.model == "XRouting":
            return np.array(edge_state1, dtype=np.float32), np.array(final_position_encoding, dtype=np.float32)
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
        print("episode:", self.episode_counter, "***********************Reward**********************:", -self.travel_time)

        return -self.travel_time

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
            if edge.id[0] != ":":
                next_edges = dict()
                turn_info[edge.id] = next_edges

        # obtain the adjacent edges of each edge
        connections = list(sumolib.xml.parse(file_name, 'net', attr_connection))[0]
        for connection in connections.connection:
            if connection.attr_from[0] != ':':
                edge = connection.attr_from
                direction = connection.dir
                if direction == "L" or direction == "R":
                    direction = "s"
                next_edge = connection.to
                turn_info[edge][direction] = next_edge

        print(turn_info)

        return turn_info
