from bidding_env import BiddingStaticEnv
#from dqn import DeepQNetwork
from dqn_keras import DQNAgent
import argparse
import sa_auction

# run episodes
def test(env, agent, args):
    total_steps = 0
    episode_cnt = env.episode_cnt
    no_reward_cnt = 0
    total_reward = 0
    for episode in range(episode_cnt):
        state  = env.reset()  
        episode_reward = 0
        steps = 0
        while True:
            ba = state[2]
            state = state.reshape(1, env.state_size)
            action_idx = agent.act(state)
            bidprice = sa_auction.get_bidprice(action_idx, ba)
            next_state, reward, done = env.step(bidprice)
            print("{} | {}".format(env.current_adgrp_name, bidprice))

            if reward == 0.0:
                no_reward_cnt +=1
            episode_reward += reward
            state = next_state  
            steps += 1
            total_steps += 1
            if done: break
 
        total_reward += episode_reward
        #if episode % 100 == 0:
        #    print("current total_reward: {}, current mean_reward: {}".format(total_reward, total_reward/total_steps))

    print('step cnt: {}, no reward_cnt: {} winning_rate: {}'.format(total_steps, no_reward_cnt, (total_steps - no_reward_cnt)/float(total_steps)))
    print('bidding over: total_reward({}), mean_reward({})'.format(total_reward, total_reward/float(total_steps)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--state_size', type=int, default=6)
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--epsilon_decay', type=float, default=0.995)
    parser.add_argument('--epsilon_min', type=float, default=0.01)
    parser.add_argument('--strategy', type=str, default='rl')

    args = parser.parse_args()

    n_action = sa_auction.action_size()

    env = BiddingStaticEnv(args.data_path, args.state_size)

    agent = DQNAgent(n_action, env.state_size, epsilon=args.epsilon,
                     epsilon_min=args.epsilon_min, epsilon_decay=args.epsilon_decay)
    agent.load(args.model_path)

    test(env, agent, args)


if __name__ == "__main__":
    main()




    
