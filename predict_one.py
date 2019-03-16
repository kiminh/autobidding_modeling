from bidding_static_env import BiddingStaticEnv
from dqn_keras import DQNAgent
import argparse
import sa_auction
import sys
import numpy as np


def get_opt_bidprice(agent):
    state = raw_input("input observation:")
    state = map(lambda n: float(n), state.split(","))
    ctr = state[-1]
    state = np.array(state)
    state = np.reshape(state, [1, agent.state_size]) 
    print(state)

    # TODO: check obs dimmesion
    action = agent.act(state)
    return sa_auction.get_bidprice(action, ctr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state_size', type=int, default=6)
    parser.add_argument('--model_path', type=str, required=True)
    args = parser.parse_args()

    agent = DQNAgent(sa_auction.action_size(), args.state_size, batch_size=1)    
    agent.load(args.model_path)

    try:
        while True:
            opt_bidprice = get_opt_bidprice(agent)
            print("opt_bidprice: {}".format(opt_bidprice))
        
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()




    
