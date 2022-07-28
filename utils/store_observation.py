# -*- coding: utf-8 -*-
# @Description: This function is used to store the observation and the corresponding action
# @author: victor
# @create time: 2022-07-27-22:30

import xlwt


def store_observation_action(observation_action_data, file_dir):
    data_length = len(observation_action_data)
    for i in range(data_length):
        agent = observation_action_data[i]
        agent_id = agent[0]["rl_id"]
        workbook = xlwt.Workbook()
        for j in range(len(agent)):
            sheet = workbook.add_sheet('observation_action_' + str(j))
            observation = agent[j]['observation']
            position_encoding = agent[j]['position_encoding']
            action = agent[j]['action']
            edge = agent[j]['edge']
            current_edge = agent[j]['current_edge']
            sorted_edges = agent[j]['sorted_edges']
            average_speed, vehicle_density, end_distance, \
            travel_time, mask, edge_angle = zip(*observation)

            # write the column name
            sheet.write(0, 0, "edge_id")
            sheet.write(0, 1, "position_encoding")
            sheet.write(0, 2, "average speed")
            sheet.write(0, 3, "vehicle density")
            sheet.write(0, 4, "end distance")
            sheet.write(0, 5, "travel time")
            sheet.write(0, 6, "mask")
            sheet.write(0, 7, "edge angle")
            sheet.write(0, 8, "action")
            sheet.write(0, 9, "edge")
            sheet.write(0, 10, "current_edge")

            # write the six observation elements
            for k in range(1, len(average_speed) + 1):
                sheet.write(k, 0, str(sorted_edges[k - 1]))
                sheet.write(k, 1, float(position_encoding[k - 1]))
                sheet.write(k, 2, float(average_speed[k - 1]))
                sheet.write(k, 3, float(vehicle_density[k - 1]))
                sheet.write(k, 4, float(end_distance[k - 1]))
                sheet.write(k, 5, float(travel_time[k - 1]))
                sheet.write(k, 6, float(mask[k - 1]))
                sheet.write(k, 7, float(edge_angle[k - 1]))

            # write the corresponding actions
            sheet.write(1, 8, int(action))
            sheet.write(1, 9, str(edge))
            sheet.write(1, 10, str(current_edge))

        workbook.save(file_dir + 'agent_' + str(agent_id) + '.xls')
