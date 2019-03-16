from bidding_env import BiddingStaticEnv
#from dqn import DeepQNetwork
from dqn_keras import DQNAgent
import argparse
import auction


# run episodes
def train(env, agent):
    total_steps = 0
    episode_cnt = env.episode_cnt
    reward_100 = 0
    total_reward = 0

    for episode in range(1,episode_cnt+1):
        state = env.reset()
        episode_reward = 0
        steps = 0
        while True:
            prev_cost = int(state.cost)
            reshaped_state = state.to_array().reshape(1, env.state_size)
            action_idx = agent.act(reshaped_state)
            bidprice = auction.get_bidprice(action_idx, prev_cost)
            #print(prev_cost, bidprice)
            next_state, reward, done = env.step(bidprice)
            agent.store_transition(state, action_idx, reward, done, next_state)
            if total_steps > agent.batch_size:
                agent.replay()
                agent.target_train()
 
            episode_reward += reward
            state = next_state
            steps += 1
            total_steps += 1
            if done: break

        total_reward += episode_reward
        reward_100 += episode_reward
        if episode % 100 == 0:
            print("Mean reward {}/ 100-meam_reward {}".format(total_reward/episode, reward_100/100.0))
            reward_100 = 0
            agent.save()
 
    print('bid-training finished')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--adgrp_dict_file', type=str, required=True)
    parser.add_argument('--state_size', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--epsilon_min', type=float, default=0.01)
    parser.add_argument('--epsilon_decay', type=float, default=0.995)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--learning_rate', type=float, default=0.0000005)
    parser.add_argument('--memory_size', type=int, default=10000) 
    parser.add_argument('--model_dir', type=str, default=".") 
    args = parser.parse_args()

    action_size = auction.action_size()

    # build env.
    env = BiddingStaticEnv(args.data_path, args.adgrp_dict_file, args.state_size)

    # init agent
    agent = DQNAgent(action_size, args.state_size, batch_size=args.batch_size, \
                     epsilon=args.epsilon, epsilon_min=args.epsilon_min, \
                     epsilon_decay=args.epsilon_decay, gamma=args.gamma, \
                     memory_size=args.memory_size, lr=args.learning_rate, \
                     model_dir=args.model_dir)

    if args.model_path:
        agent.load(args.model_path)

    # train 
    train(env, agent)


if __name__ == "__main__":
    main()




    
