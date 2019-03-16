import os
import sys
import numpy
import pandas as pd
import argparse
import math
import collections

BID_COUNT = 1
BUDGET_MIN = 50
BUDGET_MAX = 1000000000
IMP_MIN = 0
IMP_MAX = 5100000
CLK_MIN = 0
CLK_MAX = 130000
COST_MIN = 0
COST_MAX = 22000000


# read auction csv file
def load_data(data_file):
    data_map = collections.defaultdict(dict)
    adgrp_id_dict = {}
    adgrp_index = 1
    max_alpha = None
    min_alpha = None
    with open(data_file) as f:
        for line in f:
            #print("line:", line)
            s_line = line.strip().split(",")
            # ymd, time, budget, bid, adgrp, cost, imp, click, ctr
            ymd = s_line[0]
            #bid = int(s_line[0])
            time = int(s_line[1])
            budget = int(s_line[2])
            bid = int(s_line[3])
            adgrp_name = s_line[4]
            cost = int(s_line[5])
            imp = int(s_line[6])
            click = int(s_line[7])
            ctr = float(s_line[8])

            if adgrp_name not in adgrp_id_dict:
                adgrp_id_dict[adgrp_name] = adgrp_index
                adgrp_index +=1

            data_adgrp_dict = data_map[adgrp_name]
            if ymd not in data_adgrp_dict:
                data_adgrp_dict[ymd] = {}

            data_adgrp_dict[ymd][time] = (budget, bid, cost, imp, click, ctr)

    return data_map, adgrp_id_dict


# generate auction list
def gen_auction_experience_list(save_dir, data_dict, adgrp_id_dict):
    auction_experience_file = os.path.join(save_dir, "episodes.txt")
    with open(auction_experience_file, "w") as f:
        for adgrp_name in data_dict.keys():
	    for ymd in data_dict[adgrp_name].keys():
                _data_dict = data_dict[adgrp_name][ymd]
                for i in range(1,BID_COUNT+1):
                    try:
                        budget, bid, cost, imp, click, ctr = _data_dict[i]
                        next_budget, next_bid, next_cost, next_imp, next_click, next_ctr = _data_dict[i+1]
                        current_state = _get_state(i, budget, cost, imp, click, ctr)
                        next_state = _get_state(i+1, next_budget, next_cost, next_imp, next_click, next_ctr)
                        #print(current_state)
                        #action = next_bid
                        action = next_cost
                        #reward = next_click/float(next_imp)*100 #next_click
                        reward = next_click
                        if reward == 0:
                            print("... 0.0",reward, next_click, next_imp)
                    
                        f.write("{}\t{}\t{}\t{}\t{}\n".format(adgrp_name, ','.join(current_state), \
                                                              str(next_bid), str(reward), ','.join(next_state)))
                    except KeyError:
                        continue
    adgrp_id_dict_file = os.path.join(save_dir, "adgrp_id_dict.txt")
    dict_str = ''
    for adgrp_name, adgrp_id in sorted(adgrp_id_dict.items(), key=lambda x: x[1]):
        dict_str += "{}:{}\n".format(adgrp_name, adgrp_id)
    
    with open(adgrp_id_dict_file, "w") as f:
        f.write(dict_str)
    

def minmax_normalization(v, min_v, max_v):
    if v >= max_v:
        return 1.0
    elif v <= min_v:
        return 0.0
    else:
        return float(v-min_v)/(max_v - min_v)


def _get_state(time, budget, cost, imp, click, ctr):
    #cost = minmax_normalization(cost, COST_MIN, COST_MAX)
    #imp = minmax_normalization(imp, IMP_MIN, IMP_MAX)
    #click = minmax_normalization(click, CLK_MIN, CLK_MAX)
    #budget = minmax_normalization(budget, BUDGET_MIN, BUDGET_MAX)
    #imp = math.log(imp)
    #click = math.log(click)
    day_ctr = float(click)/float(imp)
 
    return [str(time), str(budget), str(cost), str(imp), str(click), str(day_ctr)] 
    #return [str(time), budget_str, cost_str, imp_str, click_str, str(ctr)]


def main():
    parser = argparse.ArgumentParser(description='data_generator')
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)

    args = parser.parse_args()

    data_file = args.data_file
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    data_dict, adgrp_id_dict = load_data(data_file)
    gen_auction_experience_list(save_dir, data_dict, adgrp_id_dict)


if __name__ == "__main__":
    main()

"""
 python data_generator.py --data_file ../data/sa_auction_data_20180820.csv  --save_dir ../data/20180820
"""
