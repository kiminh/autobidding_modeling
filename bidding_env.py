import os
import random
import math
import collections
import auction
import numpy as np

class BiddingStaticEnv:
    # data_path: data file path
    # n_state : state size
    # n_feature : feature size(cnt)
    def __init__(self, data_path, adgrp_id_dict_file, state_size):
        self.data_path = data_path        
        self.last_time = 2
	self.experience_index = 0
	self.episode_index = 0
        self.episode_cnt = 0
        self.episodes = []
        self.experiences = []
        self.state_size = state_size
        self.winning_price_index = 2 # cost is winning price
        self.current_adgrp_name = None
        self.adgrp_id_dict = self._load_adgrp_id_dict(adgrp_id_dict_file)
        self._load_episodes()

    def _load_adgrp_id_dict(self, adgrp_id_dict_file):
        adgrp_id_dict = collections.defaultdict(lambda:0)

        with open(adgrp_id_dict_file) as f:
            for line in f:
                pair_line = line.split(':')
                adgrp_id_dict[pair_line[0]] = int(pair_line[1])

        return adgrp_id_dict

    def _load_episodes(self):
        self.episodes = []
        self.episode_cnt = 0
        with open(self.data_path) as f:
            experiences = []
            for line in f:
                # experience : adgrp_name, state, action, reward, next_state
                ex = line.strip().split()
                adgrp_name = ex[0]
                adgrp_id = self.adgrp_id_dict[adgrp_name]

                #state : time, budget, cost, day_ctr, click, mon_ctr
                state = map(lambda d: float(d), ex[1].split(","))
                state = State(adgrp_id, *state)

                #assert len(state) == self.state_size
                ctr = state.ctr
                action = float(ex[2])
                reward = float(ex[3])
                next_state =  map(lambda d: float(d), ex[4].split(","))
                next_state = State(adgrp_id, *next_state)

                experience = Experience(state, action, reward, next_state)
                experiences.append(experience)
                if next_state.time == self.last_time:
                    experience.done = True
                    self.episodes.append((adgrp_name, experiences))
                    self.episode_cnt += 1
                    experiences = []

            print("loaded {} episode".format(self.episode_cnt))
            
    def reset(self):
        self.current_adgrp_name = self.episodes[self.episode_index][0]
        self.experiences = self.episodes[self.episode_index][1]
        self.episode_index = (self.episode_index + 1) % self.episode_cnt
        self.experience_index = 0
        observation = self.experiences[self.experience_index]
        start_state = observation.state
        return start_state
            
    def step(self, bidprice):
        if self.experience_index >= len(self.experiences):
            raise ValueError("no experience in the episode")
        
        ex = self.experiences[self.experience_index]
        done = False
        self.experience_index +=1

        next_state, reward = ex.next_state, ex.reward
        winning_price = int(next_state.cost)
        t = next_state.time
        if t == self.last_time:
            done = True
        #bidprice = auction.get_bidprice(action_idx, ctr)
        if bidprice and bidprice < winning_price:
            reward = 0.0

        return next_state, reward, done

    @property
    def states(self):
        pass

    @property
    def actions(self):
        pass


class State:
    def __init__(self, adgrp_id, time, budget, cost, imp, click, ctr):
        self.adgrp_id = adgrp_id
        self.time = time
        self.budget = budget
        self.cost = cost
        self.imp = imp
        self.click = click
        self.ctr = ctr

    def to_array(self):
        return np.array([self.time, self.budget, self.cost, self.ctr, 
                         self.click, self.ctr])


class Experience:
    def __init__(self, state, action, reward, next_state):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
	

def main():
    bidding = BiddingStaticEnv("../data/20180901/episodes.txt", 5, 1000)
    episode_cnt = 0
    while episode_cnt < 10:
        print("episode {}".format(episode_cnt))
        obs = bidding.reset()
        for _ in range(1):
            print(bidding.step(None))

        print("")
        episode_cnt += 1
 
if __name__ == "__main__":
    main()
