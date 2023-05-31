import argparse
import gym
import os
import sys
import pickle
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
from core.common import estimate_advantages
from core.agent import Agent
from models.rnn_policy import RnnPolicyNet
from models.rnn_value import RnnValueNet
from core.ppo import ppo_step
import assembly_gymenv

import pandas as pd
import numpy as np
from torch.nn.functional import normalize as t_norm

from contextlib import redirect_stdout
HERE = os.path.dirname(__file__)
LOG_PATH = os.path.join(HERE, "../", "data/")

HERE = os.path.dirname(__file__)
DATA_PATH = os.path.join(HERE, "../", "data/")


# min_batch_size = 512
eval_batch_size = 4
max_iter_num = 2000
log_interval = 1

parser = argparse.ArgumentParser(description='PyTorch A2C example')
parser.add_argument('--log-std', type=float, default=-1.5, metavar='G',
                    help='log std for the policy (default: -0.0)')
parser.add_argument('--hidden-n', type=int, default=8, metavar='G',
                    help='number of hidden neurons in RNN (default: 64)')
parser.add_argument('--hidden-l', type=int, default=1, metavar='G',
                    help='number of hidden layers in RNN (default: 2)')
parser.add_argument('--gamma', type=float, default=1, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--num-threads', type=int, default=4, metavar='N',
                    help='number of threads for agent (default: 4)')
parser.add_argument('--min-batch-size', type=int, default=1024, metavar='N',
                    help='minimal batch size per A2C update (default: 2048)')
parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                    help='gae (default: 0.95)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--save-model-interval', type=int, default=0, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
parser.add_argument('--clip-epsilon', type=float, default=0.2, metavar='N',
                    help='clipping epsilon for PPO')
parser.add_argument('--seed', type=int, default=1234, metavar='N',
                    help='random seed')
args = parser.parse_args()

dtype = torch.float32
torch.set_default_dtype(dtype)
device = torch.device('cpu')

"""environment"""
env = gym.make('assembly_gymenv/AssemblyGymEnv-v0')
state_dim = env.observation_space.shape
is_disc_action = len(env.action_space.shape) == 0
running_state = None
# running_state = ZFilter((state_dim[0], state_dim[1]), clip=5)
# running_reward = ZFilter((1,), demean=False, clip=10)

"""seeding"""
seed = args.seed #[28,7,384,93,571,3947,39,102]
np.random.seed(seed)
torch.manual_seed(seed)
env.seed(seed)

"""define actor and critic"""
# combine the policy and value net into one
policy_net = RnnPolicyNet(env.action_space.shape[0], 
                          hidden_n=args.hidden_n, 
                          hidden_l=args.hidden_l,
                          log_std=args.log_std)
value_net = RnnValueNet(hidden_n=args.hidden_n, 
                        hidden_l=args.hidden_l)

policy_net.to(device)
value_net.to(device)

optimizer_policy = torch.optim.SGD(policy_net.parameters(), lr=0.002, momentum=0.5)
optimizer_value = torch.optim.SGD(value_net.parameters(), lr=0.002, momentum=0.5)

# optimization epoch number and batch size for PPO
optim_epochs = 1
optim_batch_size = args.min_batch_size

"""create agent"""
agent = Agent(env, policy_net, device, running_state=running_state, num_threads=args.num_threads)


def update_params(batch):

    # states = torch.from_numpy(np.stack(batch.state)).to(dtype).to(device)
    states = [torch.from_numpy(item).to(dtype).to(device) for item in batch.state]
    actions = torch.from_numpy(np.stack(batch.action)).to(dtype).to(device)
    rewards = torch.from_numpy(np.stack(batch.reward)).to(dtype).to(device)
    masks = torch.from_numpy(np.stack(batch.mask)).to(dtype).to(device)
    # print("Rewards: ")
    # print(rewards)

    with torch.no_grad():
        values = value_net(states)
        fixed_log_probs = policy_net.get_log_prob(states, actions)

    """get advantage estimation from the trajectories"""
    advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, args.tau, device)

    """perform mini-batch PPO update"""
    optim_iter_num = int(math.ceil(len(states) / optim_batch_size))
    for _ in range(optim_epochs):
        perm = np.arange(len(states))
        np.random.shuffle(perm)
        perm = LongTensor(perm).to(device)

        perm_states = [states[idx] for idx in perm]
        actions, returns, advantages, fixed_log_probs = \
            actions[perm].clone(), returns[perm].clone(), advantages[perm].clone(), fixed_log_probs[perm].clone()

        for i in range(optim_iter_num):
            ind = slice(i * optim_batch_size, min((i + 1) * optim_batch_size, len(states)))
            states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = \
                perm_states[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind]

            ppo_step(policy_net, value_net, optimizer_policy, optimizer_value, 1, states_b, actions_b, returns_b,
                     advantages_b, fixed_log_probs_b, args.clip_epsilon, args.l2_reg)


def main():

    hist_reward = []
    hist_maxheight = []
    hist_pos = []

    for i_iter in range(max_iter_num):

        """generate multiple trajectories that reach the minimum batch_size"""
        batch, log = agent.collect_samples(args.min_batch_size, render=False)
        
        t0 = time.time()
        update_params(batch)
        t1 = time.time()

        with open(LOG_PATH+'log.txt', 'a') as f:
            with redirect_stdout(f):
                print(''.join(['=']*40) + ' Evaluation '+ ''.join(['=']*40))
        """evaluate with determinstic action (remove noise for exploration)"""
        _, log_eval = agent.collect_samples(eval_batch_size, mean_action=True)
        t2 = time.time()

        print('{}\tT_sample {:.4f}\tT_update {:.4f}\tT_eval {:.4f}\ttrain_R_min {:.2f}\ttrain_R_max {:.2f}\ttrain_R_avg {:.2f}\teval_R_avg {:.2f}'.format(
                i_iter, log['sample_time'], t1-t0, t2-t1, log['min_reward'], log['max_reward'], log['avg_reward'], log_eval['avg_reward']))

        hist_reward.append({'avg': log.get('avg_reward'),
                            'min': log.get('min_reward'),
                            'max': log.get('max_reward'),
                            'eval_avg': log_eval['avg_reward']
                            })
        # hist_maxheight.append(log.get('max_height'))
        # hist_pos.append(log.get('hist_pos'))

        if i_iter % 10 == 0:
            pd.DataFrame.from_records(hist_reward).to_csv(DATA_PATH + 'reward_{}_{}.csv'.format(seed, i_iter))
            # pd.DataFrame.from_records(hist_reward).to_csv(DATA_PATH + 'reward.csv', mode='a')
            # pd.Series(hist_maxheight).to_csv(DATA_PATH + 'height.csv', mode='a')
            # pd.Series(hist_pos).to_csv(DATA_PATH + 'pose.csv', mode='a')
            # hist_reward = []
            # hist_maxheight = []
            # hist_pos = []

        # if save_model_interval > 0 and (i_iter+1) % save_model_interval == 0:
        #     to_device(torch.device('cpu'), policy_net, value_net)
        #     pickle.dump((policy_net, value_net, running_state),
        #                 open(os.path.join(assets_dir(), 'learned_models/assembly_a2c.p'), 'wb'))
        #     to_device(device, policy_net, value_net)

        """clean up gpu memory"""
        torch.cuda.empty_cache()


if __name__=="__main__":
    main()