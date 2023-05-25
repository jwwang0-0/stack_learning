import argparse
import gym
import os
import sys
import pickle
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
from models.cnn_policy import CnnLabelPolicy
from models.cnn_value import CnnLabelValue
from core.a2c import a2c_label_step
from core.common import estimate_advantages
from core.agent import Agent
import assembly_gymenv

import pandas as pd
import numpy as np


HERE = os.path.dirname(__file__)
DATA_PATH = os.path.join(HERE, "../", "data/")


min_batch_size = 32
eval_batch_size = 8
max_iter_num = 2000
log_interval = 8

parser = argparse.ArgumentParser(description='PyTorch A2C example')
parser.add_argument('--log-std', type=float, default=-2.0, metavar='G',
                    help='log std for the policy (default: -0.0)')
parser.add_argument('--gamma', type=float, default=0.9, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                    help='gae (default: 0.95)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--save-model-interval', type=int, default=0, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
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
seed = 2345
np.random.seed(seed)
torch.manual_seed(seed)
env.seed(seed)

"""define actor and critic"""
# combine the policy and value net into one
policy_net = CnnLabelPolicy(env.action_space.shape[0], log_std=args.log_std)
value_net = CnnLabelValue()

policy_net.to(device)
value_net.to(device)

optimizer_policy = torch.optim.SGD(policy_net.parameters(), lr=0.001)
optimizer_value = torch.optim.SGD(value_net.parameters(), lr=0.001)

"""create agent"""
agent = Agent(env, policy_net, device, running_state=running_state, num_threads=1)


def update_params(batch):

    state_img = [item.get('image') for item in batch.state]
    state_img = torch.from_numpy(np.stack(state_img)).to(dtype).to(device)
    actions = torch.from_numpy(np.stack(batch.action)).to(dtype).to(device)
    rewards = torch.from_numpy(np.stack(batch.reward)).to(dtype).to(device)
    masks = torch.from_numpy(np.stack(batch.mask)).to(dtype).to(device)
    print("Rewards: ")
    print(rewards)

    with torch.no_grad():
        state_img = state_img.view((-1, 1, 1000, 1000))
        values, _ = value_net(state_img)

    """get advantage estimation from the trajectories"""
    advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, args.tau, device)

    label_img = [1 if item.get('label_instable') is True else 0 
                 for item in batch.state]
    label_img = torch.tensor(label_img).to(dtype)

    """perform TRPO update"""
    a2c_label_step(policy_net, value_net, 
                   optimizer_policy, optimizer_value, 
                   label_img, state_img, 
                   actions, returns, advantages, args.l2_reg)



def main():

    hist_reward = []
    hist_maxheight = []
    hist_pos = []

    for i_iter in range(max_iter_num):

        """generate multiple trajectories that reach the minimum batch_size"""
        batch, log = agent.collect_samples(min_batch_size, render=False)

        t0 = time.time()
        update_params(batch)
        t1 = time.time()

        hist_reward.append({'avg': log.get('avg_reward'),
                            'min': log.get('min_reward'),
                            'max': log.get('max_reward'),
                            })
        hist_maxheight.append(log.get('max_height'))
        hist_pos.append(log.get('hist_pos'))

        if i_iter % log_interval == 0:
            pd.DataFrame.from_records(hist_reward).to_csv(DATA_PATH + 'reward.csv', mode='a')
            pd.Series(hist_maxheight).to_csv(DATA_PATH + 'height.csv', mode='a')
            pd.Series(hist_pos).to_csv(DATA_PATH + 'pose.csv', mode='a')

        #     """evaluate with determinstic action (remove noise for exploration)"""
        #     _, log_eval = agent.collect_samples(eval_batch_size, mean_action=True)
        #     t2 = time.time()

        #     print('{}\tT_sample {:.4f}\tT_update {:.4f}\tT_eval {:.4f}\ttrain_R_min {:.2f}\ttrain_R_max {:.2f}\ttrain_R_avg {:.2f}\teval_R_avg {:.2f}'.format(
        #         i_iter, log['sample_time'], t1-t0, t2-t1, log['min_reward'], log['max_reward'], log['avg_reward'], log_eval['avg_reward']))

        # if save_model_interval > 0 and (i_iter+1) % save_model_interval == 0:
        #     to_device(torch.device('cpu'), policy_net, value_net)
        #     pickle.dump((policy_net, value_net, running_state),
        #                 open(os.path.join(assets_dir(), 'learned_models/assembly_a2c.p'), 'wb'))
        #     to_device(device, policy_net, value_net)

        """clean up gpu memory"""
        torch.cuda.empty_cache()


if __name__=="__main__":
    main()