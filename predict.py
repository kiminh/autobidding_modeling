from bidding_env import BiddingStaticEnv
#from dqn import DeepQNetwork
from dqn_keras import DQNAgent
import argparse
import auction

AVG_ADGRP_CTR = 0.004
AVG_BIDPRICE =  1400

# run episodes
def predict(env, agent, args):
    total_steps = 0
    episode_cnt = env.episode_cnt
    reward_cnt = 0
    total_reward = 0
    results = []
    for episode in range(episode_cnt):
        state  = env.reset()  
        episode_reward = 0
        steps = 0
        while True:
            if args.strategy == 'lin':
                print(state.ctr)
                bidprice = AVG_BIDPRICE * (state.ctr/AVG_ADGRP_CTR)
            else:
                prev_cost = state.cost
                reshaped_state = state.to_array().reshape(1, env.state_size)
                action_idx = agent.act(reshaped_state, args.strategy)
                bidprice = auction.get_bidprice(action_idx, prev_cost)

            next_state, reward, done = env.step(bidprice)
            results.append("{}\t{}".format(env.current_adgrp_name, int(bidprice)))
            print(next_state.cost, bidprice, state.cost)
            if reward > 0.0:
                reward_cnt +=1
            episode_reward += reward
            state = next_state  
            steps += 1
            total_steps += 1
            if done: break
 
        total_reward += episode_reward
        #if episode % 100 == 0:
        #    print("current total_reward: {}, current mean_reward: {}".format(total_reward, total_reward/total_steps))
        
    if args.mode == 'test':
        print('step cnt: {}, reward_cnt: {} winning_rate: {}'.format(total_steps, reward_cnt, reward_cnt/float(total_steps)))
        print('bidding over: total_reward({}), mean_reward({})'.format(total_reward, total_reward/float(total_steps)))
 
    return results


def save_bidprice(path, results):
    with open(path, 'w') as f:
        f.write('\n'.join(results))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--adgrp_dict_file', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--state_size', type=int, default=6)
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--epsilon_decay', type=float, default=0.995)
    parser.add_argument('--epsilon_min', type=float, default=0.01)
    parser.add_argument('--strategy', type=str, default='rl')
    parser.add_argument('--mode', type=str, default='predict')

    args = parser.parse_args()

    n_action = auction.action_size()

    # build env
    env = BiddingStaticEnv(args.data_path, args.adgrp_dict_file, args.state_size)

    # init agent
    agent = DQNAgent(n_action, env.state_size, epsilon=args.epsilon, 
                     epsilon_min=args.epsilon_min, epsilon_decay=args.epsilon_decay)
    agent.load(args.model_path)

    # predict bidprice
    results = predict(env, agent, args)

    # save predicted bidprice to output_path
    save_bidprice(args.output_path, results)


if __name__ == "__main__":
    main()




    
