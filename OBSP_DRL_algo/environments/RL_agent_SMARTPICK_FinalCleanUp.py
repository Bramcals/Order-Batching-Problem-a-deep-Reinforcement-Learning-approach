"""
Created on Thu Sep  5 15:55:08 2019

@author: nlbcal
"""

import pandas as pd
import random
import math
import numpy as np
import statistics
import time
import socket
import binascii
import webbrowser
import cProfile
import matplotlib.pyplot as plt
import datetime
import gym
import os

from sklearn.model_selection import train_test_split
from statistics import mean
from collections import deque
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines import DQN


class SMARTPICK(gym.Env):
    def __init__(self, config=None):
        self.config = config = config['environment']
        Data_input_train = pd.read_excel("Data_input_cutofftimes_train.xlsx")
        Data_input_test = pd.read_excel("Data_input_cutofftimes_train.xlsx")
        
        #Training or testing
        #Data_input = Data_input_train
        Data_input = Data_input_test
        
        observation_space = config['observation_space']
        action_space = config['action_space']
        reward_version_config = config['reward_version']
        reward_cap_config = config['reward_cap']
        reward_type_config = config['reward_type']
        self.penalty_action_30 = config['penalty_action_30']
        self.penalty_incorrect_action = config['penalty_incorrect_action']
        self.reward_correct_action = config['reward_correct_action']
        self.Data_input = Data_input
        self.earliest_cut_off_hour = 20
        self.earliest_cut_off_time = self.earliest_cut_off_hour * 60 * 60
        self.latest_cut_off_hour = 24
        self.latest_cut_off_time = self.latest_cut_off_hour * 60 * 60
        self.end_time_simulation = 24 * 60 * 60
        self.end_hour_simulation = 24
        self.start_time_simulation = 19 * 60 * 60
        self.start_hour_simulation = 19
        self.latest_order_arrival_hour = 23
        self.limit = 19.85*60*60
        self.simulation_time = 0
        self.plan_period = 0.25
        self.cut_off_gap = 0.25
        self.min_interval_cuf_off_order_in_time = self.cut_off_gap 
        self.max_interval_cuf_off_order_in_time = 1 * 60 * 60
       
        self.average_throughput1, self.average_throughput2, self.average_throughput3, self.average_throughput4, self.average_throughput5, self.average_throughput6 = (0,0,0,0,0,0)
        self.actions1, self.actions2, self.actions3, self.actions4, self.actions5, self.actions6, self.actions7, self.actions8, self.actions9, self.actions10, self.actions11, self.actions12, self.actions13, self.actions14, self.actions15, self.actions16, self.actions17, self.actions18, self.actions19, self.actions20, self.actions21, self.actions22, self.actions23, self.actions24, self.actions25, self.actions26, self.actions27, self.actions28,self.actions29, self.actions30, self.actions31 = np.zeros(31)
        self.zactionepisode0 , self.zactionepisode1 ,  self.zactionepisode2,  self.zactionepisode3, self.zactionepisode4, self.zactionepisode5, self.zactionepisode6, self.zactionepisode7, self.zactionepisode8, self.zactionepisode9, self.zactionepisode10, self.zactionepisode11, self.zactionepisode12, self.zactionepisode13, self.zactionepisode14, self.zactionepisode15, self.zactionepisode16, self.zactionepisode17, self.zactionepisode18, self.zactionepisode19, self.zactionepisode20, self.zactionepisode21, self.zactionepisode22, self.zactionepisode23, self.zactionepisode24, self.zactionepisode25, self.zactionepisode26, self.zactionepisode27, self.zactionepisode28, self.zactionepisode29, self.zactionepisode30 = np.zeros(31)
        self.total_actions_episode = 0
        self.total_actions = np.zeros(action_space)
        self.maxstep = 1500
        self.late_percentage = 0
        self.utilpickers = 0
        self.utilshuttles = 0
        self.utildto = 0
        self.utilsto = 0
        self.utilpacking = 0
        self.tries = 0
        self.Nstep = 0
        self.episode = 0
        self.TimeStep = 0
        

        self.tot_complete = []
        self.start_Cap = []
        self.list_late_percentage = []
        self.list_rewards = []
        self.list_episodes= []
        self.utilizations = []
        
        self.max_items_per_tote = 10
        self.time_to_pick_one_item = 24  # seconds
        self.time_to_transfer_TSU_to_GtP = 120  # seconds
        self.setup_time_ptg = 60
        self.Percentage_PtG_SIO = 0.7
        self.max_batchsize_SIO = 10
        self.max_batchsize_MIO = 12
        self.max_number_items_per_picker = 40
        self.max_state_repres_number = config['max_state_repres_number']
        self.max_number_orders_too_late = config['max_number_order_too_late']
        self.penalty_for_late_order = config['penalty_late_orders']
        self.number_of_orders_per_hour = config['NOrders_hour']
        self.total_orders = (
            self.latest_order_arrival_hour - self.start_hour_simulation
        ) * self.number_of_orders_per_hour
        self.reward_version = reward_version_config
        self.cap_reward = reward_cap_config
        self.reward_type = reward_type_config
        self.First_time_right_predictions = 0
        self.Non_30actions_rate = 0
        self.Nfaulty_actions_episode = 0

        self.shifts = [
            self.start_hour_simulation + (self.plan_period * (i))
            for i in range(
                int(
                    (self.latest_order_arrival_hour - self.start_hour_simulation)
                    / self.plan_period
                )
                + 1
            )
        ]
        self.cut_off_hours = [
            self.earliest_cut_off_hour + (self.cut_off_gap * (i))
            for i in range(
                int(
                    (self.latest_cut_off_hour - self.earliest_cut_off_hour)
                    / self.cut_off_gap
                )
                + 1
            )
        ]

        self.constraint = [
            [16],
            [17, 18],
            [16],
            [17, 18],
            [16, 17, 19],
            [16, 20],
            [17, 18],
            [16, 17, 19],
            [17, 19],
            [16, 17, 19],
        ]
        self.constraints = []

        for i in range(0, len(self.constraint)):
            self.constraints.append(self.constraint[i])
            self.constraints.append(self.constraint[i])
            self.constraints.append(self.constraint[i])

        comps = ["SIO", "MIO"]
        locs = ["PtG", "GtP", "Both"]
        tards = [0, 1, 2]
        index = 0
        self.order_types = []
        for i in comps:
            for j in locs:
                for k in tards:
                    if i == "SIO" and j == "Both":
                        index = index
                    else:
                        self.order_types.append([i, j, k])

        self.Percentage_GtP_SIO = 1 - self.Percentage_PtG_SIO
        self.Percentage_Both = self.Percentage_PtG_SIO * self.Percentage_GtP_SIO
        self.Percentage_PtG_MIO = self.Percentage_PtG_SIO / (
            self.Percentage_PtG_SIO + self.Percentage_GtP_SIO + self.Percentage_Both
        )
        self.Percentage_GtP_MIO = self.Percentage_GtP_SIO / (
            self.Percentage_PtG_SIO + self.Percentage_GtP_SIO + self.Percentage_Both
        )
        self.Percentage_Both_MIO = self.Percentage_Both / (
            self.Percentage_PtG_SIO + self.Percentage_GtP_SIO + self.Percentage_Both
        )
        self.observation_space = gym.spaces.Box(
            low=0, high=self.max_state_repres_number, shape=(1, self.config['observation_space']), dtype=np.uint8
        )
        self.action_space = gym.spaces.Discrete(self.config['action_space'])
        self.BatchID = 0

        #Create connection with automod
        self.create_connection()
        self.send_function("hold")

    def render(self): #render function used for printing
        print(self.state)


    def reset(self): #Reset function to initialize state
        self.Nstep = 0
        self.epReward = 0
        self.Nfaulty_actions = 0
        self.episode_actions = np.zeros(31)
        self.episode += 1
        self.old_penalty_orders = 0
        self.reward_next_action = 0
        self.false_prediction = 0
        self.executedactions = 0
        self.Num_late_orders_processed = [0]
        self.completed_orders = []
        self.completed_batchid = []
        self.completed_Endtime = []
        self.completed_Order_arrival_time = []
        self.action_sequence = []
        self.action_sequencepr = []
        self.utilizations = []
        self.throughput1 = []  
        self.throughput2 = []
        self.throughput3 = []
        self.throughput4 = []
        self.throughput5 = []
        self.throughput6 = []
        self.available_shuttles = 8
        self.available_pickers = 5
        self.triesbool = False
       

        self.average_order_state = []
        self.Route_ID = []
        self.tot_numbers_orders_automod = []

        Day_Data, shiftdata1, shiftdata2, shiftdata3, shiftdata4, shiftdata5 = self.Day_Data_generator(
            self.Data_input
        )
        #Choose a random hour for processing
        self.Ndataset = random.randrange(5)
        Liststarttimes = [19, 20, 21, 22, 23]
        self.start_time_simulation = Liststarttimes[self.Ndataset] * 60 * 60 
        Listdataset = [shiftdata1, shiftdata2, shiftdata3, shiftdata4, shiftdata5]
        self.Day_Data = Listdataset[self.Ndataset]

        self.Number_orders_timeperiod = len(self.Day_Data)
        self.current_time = self.start_time_simulation

        state = self.generate_state(
            self.Day_Data, self.start_time_simulation, self.start_Cap, [0]
        )
        self.state = np.reshape(state, [1, self.config['observation_space']])
        return self.state

    def generate_state(self, Data, current_time, cap, new_late_orders): 
        #Genrate state
        #Check if new orders enter the time
        #Check if orders that not have been processed, have become tardy
        #Determine the current remaining orders for processing

        self.current_orderlist = self.Day_Data.loc[
            (Data["Tot_seconds"] <= self.current_time)
            & (self.Day_Data["Processed"] == 0)
        ]
        self.current_orderlist = self.current_orderlist.sort_values(
            ["Comp", "cut_off_times"], ascending=[False, True]
        )
        tard_cat = [
            3
            if self.current_orderlist.iloc[i, 5] - self.current_time <= 0
            else 2
            if self.current_orderlist.iloc[i, 5] - self.current_time <= (15 * 60)
            else 1
            if self.current_orderlist.iloc[i, 5] - current_time <= (38 * 60)
            else 0
            for i in range(len(self.current_orderlist))
        ]
        self.current_orderlist["Tard_cat"] = tard_cat
        state = [
            len(
                self.current_orderlist.loc[
                    (self.current_orderlist["Comp"] == i[0])
                    & (self.current_orderlist["Stor_loc"] == i[1])
                    & (self.current_orderlist["Tard_cat"] == i[2])
                ]
            )
            for i in self.order_types
        ]
        self.late_non_processed_orders = len(
            self.current_orderlist[self.current_orderlist["Tard_cat"] == 3]
        )

        self.late_orders = [0]
        self.late_orders[0] = self.late_non_processed_orders + new_late_orders[0]
        self.late_orders[0] = (self.late_orders[0]/self.max_number_orders_too_late) * self.max_state_repres_number
        timeindication = [current_time - self.start_time_simulation] 
        timeindication[0] = (timeindication[0] /3600) * self.max_state_repres_number

        Norders = [(sum(self.tot_numbers_orders_automod) / self.Number_orders_timeperiod) * self.max_state_repres_number]

        if cap[0] > 0:
            cap[0] = self.max_state_repres_number
        if cap[1] > 0:
            cap[1] = self.max_state_repres_number
        
        state = state + self.late_orders + cap + timeindication + Norders

        #Denormalize state
        for i in range(len(state)):
            if state[i] > self.max_state_repres_number:
                state[i] = self.max_state_repres_number
        next_state = np.reshape(state[0:23], [1, 23])
        return next_state

    def create_connection(self): 
        #Create connection with simulation model via sockets. Python code is the server, automod(simulation model) is the client
        HOST = "localhost"
        PORT = 1235
        
        print(socket.gethostname())
        try:
            self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        except socket.error:
            print("Bind failed. Error Code : ")
        self.s.bind((HOST, PORT))
        self.s.listen(5)
        print("Socket Listening")
        
        self.conn, self.addr = self.s.accept()
        data = self.conn.recv(1024)
        print("Connection from Automod to python established")

        decoded_message = data.decode(encoding="UTF-8")
        print(decoded_message)
        self.send_function("Connection from python to Automod established")

        sending_message = (
            "start_"
            + str(self.start_time_simulation)
            + "_"
            + str(self.end_hour_simulation * 60 * 60)
        )
        self.send_function(sending_message)

        data = self.conn.recv(1024)
        decoded_message = data.decode(encoding="UTF-8")
        delimmited_message = decoded_message.split("_")
        start_Cap = [
            int(delimmited_message[i + 2]) for i in range(len(delimmited_message[2:7]))
        ]
        self.start_Cap = start_Cap
        self.available_pickers = self.start_Cap[0]
        self.available_shuttles = self.start_Cap[1]
        self.startpickers = self.start_Cap[0]
        self.startshuttles = self.start_Cap[1]


    def disconnect(self):
        # After training the connection with the simulation model is disconnected
        self.send_function("Final")
        self.s.close()

    def send_function(self, sending_message):
        # For sending messages to Automod
        # Messages are decoded in utf8 hex 
        hex_message = sending_message.encode("utf-8").hex()
        hex_message = "02" + hex_message[0:] + hex_message[:0] + "03"
        binary = binascii.a2b_hex(hex_message)
        self.conn.send(binary)

    def step(self, action):
        #Main method to perform actions
        self.Nstep += 1
        self.TimeStep += 1  
        late_orderss = 0
        reward_orders = 0
        self.action_sequencepr.append(action)
        reward = 0 + self.reward_next_action
        penalty = 0
        self.earned_reward_correct_action = 0
        complete = False
        terminal = False
        nothing  = False

        new_action = action

        if action < 30:
            #Check feasibility of action
            NTSU = self.get_NTSU(action)
            cap = self.capacity_check(action, self.state, NTSU)
            correct_action = self.check_order_availability(self.state, action)
            new_action = action
            if cap == False or correct_action == False:
                #If action is not feasible, action is changed and nothing happens
                self.false_prediction += 1
                self.Nfaulty_actions_episode += 1
                new_action = 30
                sending_message = "nothing"
                self.tries +=  1
                nothing = True
    
        else:
            cap = False
        if action != 30:
            if cap == True and correct_action == True:
                self.executedactions+=1
                self.earned_reward_correct_action = self.reward_correct_action
           

        if new_action == 30 and nothing == False:
            sending_message = "wait"
            if self.current_time < self.start_time_simulation + 1:
                sending_message = "hold"

        if new_action < 30:
            sending_message = "action_"
            if self.current_time < self.start_time_simulation + 1:
                sending_message = "beginning_"
            Num_Orders, cut_off_time, next_batch_ID, Order_ID, NTSU, route_ID, NItems, NPickers, timeforPtG = self.get_attributes(
                new_action
            )
            self.tot_numbers_orders_automod.append(Num_Orders)
            if sending_message == "beginning_":
                sending_message = (
                    sending_message
                    + str(cut_off_time)
                    + "_"
                    + str(next_batch_ID)
                    + "_"
                    + str(NPickers)
                    + "_"
                    + str(NTSU)
                    + "_"
                    + str(route_ID)
                    + "_"
                    + str(timeforPtG)
                    + "_"
                    + str(NItems)
                    + "_"
                    + str(Num_Orders)
                    + "_"
                    + str(self.start_time_simulation)
            )
            else:
                sending_message = (
                    sending_message
                    + str(cut_off_time)
                    + "_"
                    + str(next_batch_ID)
                    + "_"
                    + str(NPickers)
                    + "_"
                    + str(NTSU)
                    + "_"
                    + str(route_ID)
                    + "_"
                    + str(timeforPtG)
                    + "_"
                    + str(NItems)
                    + "_"
                    + str(Num_Orders))
            
            

                
        self.total_actions[new_action]+=1
        self.episode_actions[new_action]+=1


        if new_action == 30: 
            i = 0
            while i<30:
                if i > 14:
                    index = i - 15
                else:
                    index = i
                
                
                if self.state[0][index] > 0:
                    NTSU = self.get_NTSU(i)
                    cap = self.capacity_check(i, self.state, NTSU) 
                    if cap == True:
                        reward = self.penalty_action_30
                        sending_message = "nothing"           
                        nothing = True
                        i +=30
                    else: 
                        reward = 0
                i+=1    
                if i == 30 and cap == False and action == 30:
                    sending_message == "wait"

        if sending_message != "nothing":
            self.tries = 0
            self.triesbool = False


            if sum(self.state[0][0:15]) == 0 and self.current_time > self.limit:
                #Terminal state if all orders have been processed and we have reached or time limit
                terminal = True
                if sum(self.state[0][0:15]): 
                    complete = True

            if sum(self.state[0][0:15]) == 0 and self.current_time < self.limit:
                # All orders are processed and we still have time left
                # Therefore a message to Automod is send such that the enivornment is simulate 
                # The simulation model provides feedback for each order that leaves the system, 
                # however the state only is provided once all orders have left the system
                sending_message= "hour"
                if terminal:
                    print("true")
            
            self.send_function(sending_message)
            if terminal == False:
                d = False
                while d == False:
                    try:
                        #Receive message from automod, either new state info or batches/orders that have been processed
                        data = self.conn.recv(1024)
                    except socket.timeout:
                        print("sleeping")
                        self.send_function("wait")
                        continue

                    if len(data) > 0: 
                        decoded_message = data.decode(encoding="UTF-8")
                        delimmited_message = decoded_message.split("_")
                        Cap_update = []

                        if delimmited_message[1] == "Dummy" or delimmited_message[0] == "Dummy":
                            if sum(self.state[0][0:15])==0:
                                self.send_function("end")
                            else: 
                                self.send_function("hold") 
                            d = False
                        elif delimmited_message[1] == "Update":
                            d = True
                            Cap_update = [
                                int(delimmited_message[i + 2])
                                for i in range(len(delimmited_message[2:7]))
                            ]
                            self.available_pickers = Cap_update[0]
                            self.available_shuttles = Cap_update[1]
                            self.simulation_time = float(delimmited_message[7])
                            self.current_time = (
                                self.start_time_simulation + self.simulation_time
                            )
                            self.utilization = [x/y for x, y in zip(Cap_update, self.start_Cap)]
                            self.utilizations.append(self.utilization)
                            self.average_utilizations = [statistics.mean(row[i] for row in self.utilizations) for i in range(len(self.utilizations[0]))]
                            

                        elif delimmited_message[1] == "BatchID":
                            if int(delimmited_message[15]) == 1:
                                self.throughput1.append(float(delimmited_message[7]))
                            if int(delimmited_message[15]) == 2:
                                self.throughput2.append(float(delimmited_message[7]))
                            if int(delimmited_message[15]) == 3:
                                self.throughput3.append(float(delimmited_message[7]))
                            if int(delimmited_message[15]) == 4:
                                self.throughput4.append(float(delimmited_message[7]))
                            if int(delimmited_message[15]) == 5:
                                self.throughput5.append(float(delimmited_message[7]))
                            if int(delimmited_message[15]) == 6:
                                self.throughput6.append(float(delimmited_message[7]))
                            if float(delimmited_message[3]) < float(delimmited_message[4]):
                                self.completed_orders.append(int(delimmited_message[6]))

                            else:
                                self.completed_orders.append((int(delimmited_message[6])))
                                penalty = penalty + (
                                    self.penalty_for_late_order
                                    * (float(delimmited_message[6]))
                                )
                                self.Num_late_orders_processed[
                                    0
                                ] = self.Num_late_orders_processed[0] + int(
                                    delimmited_message[6]
                                )
                             

                            d = False
                            sending_message = "Completed batch received"
                            if (
                                delimmited_message[8] == self.start_Cap[0]
                                and delimmited_message[9] == self.start_Cap[1]
                                and delimmited_message[10] == self.start_Cap[2]
                            ):
                                d = True
                                Cap_update = [
                                    int(delimmited_message[i + 9])
                                    for i in range(len(delimmited_message[2:7]))
                                ]
                            self.send_function(sending_message)

            else:
                Cap_update = list(self.state[0][16:])
            next_state = self.generate_state(
                self.Day_Data, self.current_time, Cap_update, self.Num_late_orders_processed
            )

            #Termination condition:
            if self.late_orders[0] >= self.max_number_orders_too_late:
                terminal = True
                complete = False

            elif sum(next_state[0][0:15]) == 0 and self.current_time > self.limit:
                terminal = True
                complete = True

            if sum(self.tot_numbers_orders_automod) >= len(self.Day_Data):
                terminal = True


            #If we have reached an end state, the simulation model will simulate until all orders have left the system
            if complete == True or terminal == True:
                self.send_function("end")
                d = False
                while d == False:
                    try:
                        data = self.conn.recv(1024)
                    except socket.timeout:
                        print("sleeping")
                        self.send_function("wait")
                        continue
                    if len(data) > 0:
                        decoded_message = data.decode(encoding="UTF-8")
                        delimmited_message = decoded_message.split("_")
                        Cap_update = []
                        if delimmited_message[1] == "Dummy":
                            self.send_function("end") 
                            d = False
                        if delimmited_message[1] == "Automod":
                            d = True
                            self.utilpickers = float(delimmited_message[10])
                            self.utilshuttles = float(delimmited_message[11])
                            self.utildto = float(delimmited_message[12])
                            self.utilsto = float(delimmited_message[13])
                            self.utilpacking = float(delimmited_message[14])
                        if delimmited_message[1] == "Update":
                            Cap_update = [
                                int(delimmited_message[i + 2])
                                for i in range(len(delimmited_message[2:7]))
                            ]
                            self.utilization = [x/y for x, y in zip(Cap_update, self.start_Cap)]
                            self.utilizations.append(self.utilization)
                            self.average_utilizations = [statistics.mean(row[i] for row in self.utilizations) for i in range(len(self.utilizations[0]))]
                            self.available_pickers = Cap_update[0]
                            self.available_shuttles = Cap_update[1]
                            
                        if delimmited_message[1] == "BatchID":
                            self.completed_batchid.append(int(delimmited_message[2]))
                            self.completed_Endtime.append(float(delimmited_message[3]))
                            self.completed_Order_arrival_time.append(
                                (delimmited_message[2:7])
                                
                            )
                            
                            if int(delimmited_message[15]) == 1:
                                self.throughput1.append(float(delimmited_message[7]))
                            if int(delimmited_message[15]) == 2:
                                self.throughput2.append(float(delimmited_message[7]))
                            if int(delimmited_message[15]) == 3:
                                self.throughput3.append(float(delimmited_message[7]))
                            if int(delimmited_message[15]) == 4:
                                self.throughput4.append(float(delimmited_message[7]))
                            if int(delimmited_message[15]) == 5:
                                self.throughput5.append(float(delimmited_message[7]))
                            if int(delimmited_message[15]) == 6:
                                self.throughput6.append(float(delimmited_message[7]))
                            
                            if float(delimmited_message[3]) < float(delimmited_message[4]):
                                self.completed_orders.append(int(delimmited_message[6]))
                            else:
                                self.completed_orders.append(int(delimmited_message[6]))
                                penalty = penalty + (
                                    self.penalty_for_late_order
                                    * (float(delimmited_message[6]))
                                )
                                self.Num_late_orders_processed[
                                    0
                                ] = self.Num_late_orders_processed[0] + int(
                                    delimmited_message[6]
                                )
                                late_orderss = late_orderss + int(
                                    delimmited_message[6])
                            sending_message = "Completed batch received"
                            self.send_function(sending_message)
                
                self.state[0][15] = self.late_orders[0]
                if self.old_penalty_orders != self.late_non_processed_orders:
                    penalty = penalty + (
                        self.late_non_processed_orders - self.old_penalty_orders
                    ) * self.penalty_for_late_order
                    self.old_penalty_orders = self.late_non_processed_orders

                if len(self.throughput1) > 1:
                    self.average_throughput1 = statistics.mean(self.throughput1)
                if len(self.throughput2) > 1:
                    self.average_throughput2 = statistics.mean(self.throughput2)
                if len(self.throughput3) > 1:
                    self.average_throughput3 = statistics.mean(self.throughput3)
                if len(self.throughput4) > 1:
                    self.average_throughput4 = statistics.mean(self.throughput4)
                if len(self.throughput5) > 1:
                    self.average_throughput5 = statistics.mean(self.throughput5)
                if len(self.throughput6) > 1:
                    self.average_throughput6 = statistics.mean(self.throughput6)                   
                
                if complete:
                    self.tot_complete.append(complete)

            if self.reward_type == 7:
                reward = reward_orders
        
            reward = reward + penalty
            
            if self.reward_type == 4 or self.reward_type == 5:
                reward = reward + self.earned_reward_correct_action
            if self.reward_version == 1 or self.reward_version == 2:
                if terminal:
                    self.late_orders[0] = self.late_orders[0] + late_orderss
                    if complete:
                        self.late_percentage = ((self.late_orders[0]) / (len(self.Day_Data)))
                    else:
                        self.late_percentage = (((self.late_orders[0]) +  (len(self.Day_Data) -  sum(self.tot_numbers_orders_automod))) / (len(self.Day_Data)))
                    self.late_percentage = (((self.late_orders[0]) +  (len(self.Day_Data) -  sum(self.tot_numbers_orders_automod))) / (len(self.Day_Data)))                   
                    self.Nfaulty_actions_episode = self.Nfaulty_actions 

                    if self.late_percentage > 1:
                        self.late_percentage = 1
                        reward = 0
                    else:
                        reward = ((1 - self.late_percentage) ** 4)
                    
                    if sum(self.tot_numbers_orders_automod) < len(self.Day_Data):
                        reward = -1
                    
                    self.list_late_percentage.append(self.late_percentage)
                    
                    self.total_actions_episode = self.Nstep
                    self.First_time_right_predictions = (self.Nstep- self.false_prediction) / self.Nstep
                    self.Non_30actions_rate = self.executedactions / self.Nstep
                    
                    #Number of actions per episode, a seperate variable is required such that tensorboard can be enabled. 
                    self.zactionepisode0 = self.episode_actions[0]
                    self.zactionepisode1 = self.episode_actions[1]
                    self.zactionepisode2 = self.episode_actions[2]
                    self.zactionepisode3 = self.episode_actions[3]
                    self.zactionepisode4 = self.episode_actions[4]
                    self.zactionepisode5 = self.episode_actions[5]
                    self.zactionepisode6 = self.episode_actions[6]
                    self.zactionepisode7 = self.episode_actions[7]
                    self.zactionepisode8 = self.episode_actions[8]
                    self.zactionepisode9  =  self.episode_actions[9]
                    self.zactionepisode10 =   self.episode_actions[10]
                    self.zactionepisode11 =   self.episode_actions[11]
                    self.zactionepisode12 =   self.episode_actions[12]
                    self.zactionepisode13 =   self.episode_actions[13]
                    self.zactionepisode14 =   self.episode_actions[14]
                    self.zactionepisode15 =   self.episode_actions[15]
                    self.zactionepisode16 =   self.episode_actions[16]
                    self.zactionepisode17 =   self.episode_actions[17]
                    self.zactionepisode18 =   self.episode_actions[18]
                    self.zactionepisode19 =   self.episode_actions[19]
                    self.zactionepisode20 =   self.episode_actions[20]
                    self.zactionepisode21 =   self.episode_actions[21]
                    self.zactionepisode22 =   self.episode_actions[22]
                    self.zactionepisode23 =   self.episode_actions[23]
                    self.zactionepisode24 =   self.episode_actions[24]
                    self.zactionepisode25 =   self.episode_actions[25]
                    self.zactionepisode26 =   self.episode_actions[26]
                    self.zactionepisode27 =   self.episode_actions[27]
                    self.zactionepisode28 =   self.episode_actions[28]
                    self.zactionepisode29 =   self.episode_actions[29]
                    self.zactionepisode30 =   self.episode_actions[30]

        #If message is nothing: do nothing and thus let the next state be the current state
        else:
            next_state = self.state
            terminal = False
            reward = self.penalty_incorrect_action

        self.total_actions[new_action]+=1
        self.episode_actions[new_action]+=1
        
        self.action_sequence.append(new_action)
        reward = reward / self.cap_reward
        self.epReward += reward

        if terminal:
            self.list_rewards.append(self.epReward)
            self.list_episodes.append(self.episode)
            #df = pd.DataFrame({'Episode':self.list_episodes,'Reward':self.list_rewards, "Late_percentage": self.list_late_percentage})
            #df.to_csv(r"C:\Users\nlbcal\OneDrive - Vanderlande\Backup-Important files\Tensorboard logs\LogDatasetx"+str(self.number_of_orders_per_hour)+".csv")
            #print("Episode: " + str(self.episode) + ", Ep Reward: " + str("{0:.2f}".format(self.epReward)) + ", Final Reward: " + str("{0:.2f}".format(reward)) +  ", Late %: " + str("{0:.2f}".format(self.late_percentage*100)) + ", N Processed Orders: " + str(sum(self.tot_numbers_orders_automod)) + ", N actions: "+ str(self.Nstep)+ ", Complete: "+ str(complete)+ ", Mean Late% 100ep: "+str("{0:.2f}".format(mean(self.list_late_percentage[-1000:])*100)))

        self.state = next_state
        return next_state, reward, terminal, {}

    def callback(self, locals_, globals_):
        #For tensorboard loggings
        self_ = locals_['self']
        # Log additional tensor
        if not self_.is_tb_set:
            with self_.graph.as_default():
                tf.summary.scalar('value_target', tf.reduce_mean(self_.value_target))
                self_.summary = tf.summary.merge_all()
            self_.is_tb_set = True
        #Log scalar value (here a random variable)
        value = np.random.random()
        summary = tf.Summary(value=[tf.Summary.Value(tag='random_value', simple_value=value)])
        locals_['writer'].add_summary(summary, self_.num_timesteps)
        return True



    def check_order_availability(self, state, action):
        #Check if orders are available
        index = action
        if action >= 15:
            index = action - 15
        if self.state[0][index] == 0:
            return False
        else:
            return True

    def get_NTSU(self, action):
        # Determine the number of NTSU/totes required for action
        if action == 30:
            return 0
        index = action
        if action >= 15:
            index = action - 15

        comp, loc, tard = self.order_types[index]
        types = self.current_orderlist.loc[
            (self.current_orderlist["Comp"] == comp)
            & (self.current_orderlist["Stor_loc"] == loc)
            & (self.current_orderlist["Tard_cat"] == tard)
        ]
        types = types.sort_values(["cut_off_times"], ascending=True)

        if len(types) == 0:
            return 99999
        if action < 15:
            NItems = types.iloc[0, 2]
            NTSU = 1
            if action > 8 and action < 12:
                NTSU = NItems
            if action >= 12:
                NItems_PtG = NItems * self.Percentage_PtG_SIO
                NItems_GtP = NItems - NItems_PtG
                NTSU = NItems_GtP + math.ceil(NItems_PtG / self.max_items_per_tote)
        elif action >= 15 and action < 21:
            batchsize = min(len(types), self.max_batchsize_SIO)
            if action < 18:
                NTSU = 1
            else:
                NTSU = min(batchsize, 5)
                if NTSU == 0:
                    NTSU = batchsize
                batchsize = NTSU
            NItems = batchsize
 
        else:
            batchsize = min(len(types), self.max_batchsize_MIO)
            if (action >= 27 and action <= 29) or (action >= 21 and action <= 23):
                max_batchsize_max_picker = 12
                batchsize = min(len(types), max_batchsize_max_picker)
            NItems = sum(types.iloc[0:batchsize, 2])
            NTSU = math.ceil(NItems / self.max_items_per_tote)
            if action >= 24 and action < 27:
                NTSU = NItems
            if action >= 27: 
                if NItems == 2:
                    NItems_PtG = 1
                else:
                    NItems_PtG = math.ceil(NItems * self.Percentage_PtG_SIO)
                NItems_GtP = NItems - NItems_PtG
                NTSU = NItems_GtP + math.ceil(NItems_PtG / self.max_items_per_tote)
        return NTSU
    
    def capacity_check(self, action, state, NTSU):
        # Depending on the action and the number of NTSU/totes
        # We determine there are resources available to proces the
        if action == 30:
            return False
        else:
            c = self.constraints[action]
            NPickers = 1
            if action <= 14:
                if len(c) == 1 and c[0] == 16 and self.available_pickers >= NPickers:
                    cap = True  # picker(s)  available
                elif (
                    len(c) == 2
                    and c[0] == 17
                    and c[1] == 18):
                    if NTSU > 3 and self.available_shuttles >= 1:
                        cap = True  # Batch Adapto and StO available
                    elif self.available_shuttles >= 1:
                        cap = True # Adapto and DtO  available
                    else:
                        cap = False
    
                elif (
                    len(c) == 3
                    and c[0] == 16
                    and c[1] == 17
                    and c[2] == 19
                    and self.available_pickers >= NPickers

                ):
                    cap = True  # Picker, Adapto and StO available
                else:  # no sufficient capacity available
                    cap = False
            elif action > 14 and action <= 29:
                if (
                    c[0] == 17
                    and c[1] == 18):
                    if NTSU > 3 and self.available_shuttles >= 1:
                        cap = True  # Batch Adapto and StO available
                    elif self.available_shuttles >= 1:
                        cap = True # Batch Adapto and DtO available
                    else:
                        cap = False
  
                elif (
                    c[0] == 16
                    and c[1] == 20
                    and self.available_pickers >= NPickers
                ):
                    cap = True  # Batch Picker and Packstation available
                elif (
                    c[0] == 17
                    and c[1] == 19):

                
                    if NTSU > 3 and self.available_shuttles >= 3:
                        cap = True  # Batch Adapto and StO available
                    elif self.available_shuttles >= 1:
                        cap = True
                    else:
                        cap = False
                elif (
                    c[0] == 16
                    and c[1] == 17
                    and c[2] == 19
                    and self.available_pickers >= NPickers

                ):
                    cap = True  # Batch Picker, Adapto and StO available
                else:  # no sufficient capacity available
                    cap = False
            return cap

    def get_attributes(self, action):
        # For sending a message to automod such that an order/batch is processed
        # This function will generate the required information; 
        # Number of orders, Number of totes, Number of pickers, Picking time in PtG, Batch ID, Route ID and cut off time
        if action == 30:
            Num_Orders = (
                cut_off_time
            ) = (
                Order_ID
            ) = NTSU = NItems = NPickers = timeforPtG = next_batch_ID = route_ID = 0
        else:
            index = action
            if action >= 15:
                index = action - 15

            comp, loc, tard = self.order_types[index]
            types = self.current_orderlist.loc[
                (self.current_orderlist["Comp"] == comp)
                & (self.current_orderlist["Stor_loc"] == loc)
                & (self.current_orderlist["Tard_cat"] == tard)
            ]
            types = types.sort_values(["cut_off_times"], ascending=True)

            if action < 15:
                Num_Orders = 1
                Order_ID = types.iloc[0, 0]
                self.BatchID = self.BatchID + 1
                next_batch_ID = self.BatchID
                self.Day_Data.loc[
                    self.Day_Data["Orderid"] == Order_ID, "Processed"
                ] = next_batch_ID
                if len(self.Day_Data) > 20:
                    self.Day_Data[self.Day_Data["Orderid"] != Order_ID]

                NItems = types.iloc[0, 2]
                NTSU = 1
                timeforPtG = (NItems * self.time_to_pick_one_item) + self.setup_time_ptg
                NPickers = 1
                if action > 8 and action < 12:
                    NTSU = NItems
                if action >= 12:
                    NItems_PtG = NItems * self.Percentage_PtG_SIO
                    NItems_GtP = NItems - NItems_PtG
                    NTSU = NItems_GtP + math.ceil(NItems_PtG / self.max_items_per_tote)
                    timeforPtG = (
                        NItems_PtG * self.time_to_pick_one_item
                    ) + self.setup_time_ptg

            elif action >= 15 and action < 21:
                # max_batchsize_SIO = 10 #SIO batch always 10 items or less
                batchsize = min(len(types), self.max_batchsize_SIO)
                Num_Orders = min(len(types), self.max_batchsize_SIO)
                
                self.BatchID = self.BatchID + 1
                next_batch_ID = self.BatchID
                
                if action < 18:
                    NTSU = 1
                    NItems = batchsize
                else:
                    NTSU = min(batchsize, self.available_shuttles)
                    if NTSU == 0:
                        NTSU = batchsize
                    batchsize = NTSU
                    Num_Orders = NTSU
                    NItems = NTSU
                NPickers = 1
                
          
                timeforPtG = (NItems * self.time_to_pick_one_item) + self.setup_time_ptg
                
                Order_ID = types.iloc[0:NItems, 0]
               
                for i in Order_ID:
                    self.Day_Data.loc[
                        self.Day_Data["Orderid"] == i, "Processed"
                    ] = next_batch_ID
                    if len(self.Day_Data) > 20:
                        self.Day_Data[self.Day_Data["Orderid"] != i]
                    else:
                        i = i

            else:
                batchsize = min(len(types), self.max_batchsize_MIO)
                if (action >= 27 and action <= 29) or (action >= 21 and action <= 23):
                    max_batchsize_max_picker = 12
                    batchsize = min(len(types), max_batchsize_max_picker)
                Num_Orders = batchsize
                Order_ID = types.iloc[0:batchsize, 0]
                self.BatchID = self.BatchID + 1
                next_batch_ID = self.BatchID
                for i in Order_ID:
                    self.Day_Data.loc[
                        self.Day_Data["Orderid"] == i, "Processed"
                    ] = next_batch_ID
                    if len(self.Day_Data) > 20:
                        self.Day_Data[self.Day_Data["Orderid"] != i]
                    else:
                        i = i

                NItems = sum(types.iloc[0:batchsize, 2])
                NTSU = math.ceil(NItems / self.max_items_per_tote)
                NPickers = 1

                if action >= 24 and action < 27:
                    NTSU = NItems
                timeforPtG = (
                    NItems / NPickers * self.time_to_pick_one_item
                ) + self.setup_time_ptg
                if action >= 27:  # this is the action where items are placed in both:
                    # Calculated how many items are placed in both and calculate number of TSU
                    if NItems == 2:
                        NItems_PtG = 1
                    else:
                        NItems_PtG = math.ceil(NItems * self.Percentage_PtG_SIO)
                    NItems_GtP = NItems - NItems_PtG
                    NTSU = NItems_GtP + math.ceil(NItems_PtG / self.max_items_per_tote)

                    timeforPtG = (
                        NItems_PtG / NPickers * self.time_to_pick_one_item
                    ) + self.setup_time_ptg
                if (action >= 18 and action < 21) or (action >= 24 and action < 27):
                    NPickers = 0

            route_IDs = [1, 5, 1, 5, 4, 2, 5, 4, 6, 4, 0]
            route_ID = route_IDs[math.trunc(action / 3)]
            if route_ID == 5 or route_ID == 6:
                timeforPtG = 0
            cut_off_time = types.iloc[0, 5]
        return (
            Num_Orders,
            cut_off_time,
            next_batch_ID,
            Order_ID,
            NTSU,
            route_ID,
            NItems,
            NPickers,
            timeforPtG,
        )

    def Day_Data_generator(self, Data):
        # Take sample of orders from data set 
        # With this sample we create cut off times between 20:00 and 24:00
        # And determine where the orders are stored, PtG or GtP. 
        percentage_NOrders_arrive_after_2300 = 0.05
        extra_orders = math.ceil(
            self.total_orders * percentage_NOrders_arrive_after_2300
        )
        sample_data = Data.sample(n=(self.total_orders + extra_orders))
        sample_data = sample_data.sort_values(["Tot_seconds"], ascending=True)


        Day_Data = sample_data.sort_values(
            ["Comp", "Tot_seconds"], ascending=[False, True]
        )
        sample_SIO = random.sample(
            range(len(Day_Data[Day_Data["Comp"] == "SIO"])),
            len(Day_Data[Day_Data["Comp"] == "SIO"]),
        )
        sample_MIO = random.sample(
            range(
                len(Day_Data[Day_Data["Comp"] == "SIO"]),
                (
                    len(Day_Data[Day_Data["Comp"] == "MIO"])
                    + (len(Day_Data[Day_Data["Comp"] == "SIO"]))
                ),
            ),
            len(Day_Data[Day_Data["Comp"] == "MIO"]),
        )

        SIO_PtG_Orders = int(self.Percentage_PtG_SIO * len(sample_SIO))
        MIO_PtG_Orders = int(self.Percentage_PtG_MIO * len(sample_MIO))
        MIO_GtP_Orders = int(self.Percentage_GtP_MIO * len(sample_MIO)) + MIO_PtG_Orders
        processed = [0] * len(Day_Data)
        stor_loc = [" "] * len(Day_Data)
        for j in range(len(sample_SIO)):
            if j < SIO_PtG_Orders:
                stor_loc[sample_SIO[j]] = "PtG"
            else:
                stor_loc[sample_SIO[j]] = "GtP"
        for j in range(len(sample_MIO)):
            if j < MIO_PtG_Orders:
                stor_loc[sample_MIO[j]] = "PtG"
            elif j < MIO_GtP_Orders:
                stor_loc[sample_MIO[j]] = "GtP"
            else:
                stor_loc[sample_MIO[j]] = "Both"

        Day_Data["Stor_loc"] = stor_loc
        Day_Data["Processed"] = processed

        List_Order_times = []
        order_time = 0
        Day_Data = Day_Data.sort_values(["cut_off_times"], ascending=True)
        for i in range(len(Day_Data)):
            List_Order_times.append(order_time + (19*60*60))
            i += 1
            if i % (self.number_of_orders_per_hour * self.plan_period) == 0:
                order_time = order_time + (self.plan_period * 60 * 60)
        Day_Data["Tot_seconds"] = List_Order_times


        #Creating cut off times for scenario B: multiple cut off time per 15 minutes
        for i in range(len(Day_Data)):
            Order_in_time = Day_Data.iloc[i,4]
            floatn = random.random()
            
            if floatn < 0.33:
                cut_off_time = 900 + Order_in_time
            elif floatn < 0.66:
                cut_off_time = 900*2 + Order_in_time
            else:
                floatn = random.random()
                if floatn < 0.5:
                    cut_off_time = 900*3 + Order_in_time
                else:
                    cut_off_time = 900*4 + Order_in_time

            Day_Data.iloc[i,5] = cut_off_time

        shiftdata1 = Day_Data.loc[Day_Data["Tot_seconds"] < (20 * 60 * 60)]
        shiftdata2 = Day_Data.loc[Day_Data["Tot_seconds"] < (21 * 60 * 60)]
        shiftdata3 = Day_Data.loc[Day_Data["Tot_seconds"] < (22 * 60 * 60)]
        shiftdata4 = Day_Data.loc[Day_Data["Tot_seconds"] < (23 * 60 * 60)]
        shiftdata5 = Day_Data.loc[Day_Data["Tot_seconds"] < (24 * 60 * 60)]
        return Day_Data, shiftdata1, shiftdata2, shiftdata3, shiftdata4, shiftdata5
