import argparse
import gym
import os
import sys
import pickle
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
from models.cnn_ac import BinaryCnnAC
from models.cnn_ac_value import BinaryCnnValue
from core.a2c import a2c_step
from core.common import estimate_advantages
from core.agent import Agent
import assembly_gymenv
import pandas as pd

HERE = os.path.dirname(__file__)
DATA_PATH = os.path.join(HERE, "../", "data/")

parser = argparse.ArgumentParser(description='PyTorch A2C example')
parser.add_argument('--env-name', default="Hopper-v2", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--model-path', metavar='G',
                    help='path of pre-trained model')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--log-std', type=float, default=-0.0, metavar='G',
                    help='log std for the policy (default: -0.0)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                    help='gae (default: 0.95)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--num-threads', type=int, default=1, metavar='N',
                    help='number of threads for agent (default: 4)')
parser.add_argument('--seed', type=int, default=333, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--min-batch-size', type=int, default=512, metavar='N',
                    help='minimal batch size per A2C update (default: 2048)')
parser.add_argument('--eval-batch-size', type=int, default=32, metavar='N',
                    help='minimal batch size for evaluation (default: 2048)')
parser.add_argument('--max-iter-num', type=int, default=5000, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 1)')
parser.add_argument('--save-model-interval', type=int, default=100, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
parser.add_argument('--gpu-index', type=int, default=0, metavar='N')
args = parser.parse_args()

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)

"""environment"""
env = gym.make('assembly_gymenv/AssemblyGymEnv-v0')
state_dim = env.observation_space.shape
is_disc_action = len(env.action_space.shape) == 0
running_state = ZFilter((state_dim[0], state_dim[1]), clip=5)
# running_reward = ZFilter((1,), demean=False, clip=10)

"""seeding"""
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.seed(args.seed)

"""define actor and critic"""
policy_net = BinaryCnnAC(env.action_space.shape[0], log_std=args.log_std)
value_net = BinaryCnnValue(env.action_space.shape[0], log_std=args.log_std)

policy_net.to(device)
value_net.to(device)

optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=0.01)
optimizer_value = torch.optim.Adam(value_net.parameters(), lr=0.01)

"""create agent"""
agent = Agent(env, policy_net, device, running_state=running_state, num_threads=args.num_threads)


def update_params(batch):
    states = torch.from_numpy(np.stack(batch.state)).to(dtype).to(device)
    actions = torch.from_numpy(np.stack(batch.action)).to(dtype).to(device)
    rewards = torch.from_numpy(np.stack(batch.reward)).to(dtype).to(device)
    masks = torch.from_numpy(np.stack(batch.mask)).to(dtype).to(device)
    print("Rewards: ")
    print(rewards)
    with torch.no_grad():
        states = states.view((-1, 1, 1000, 1000))
        values = value_net(states)

    """get advantage estimation from the trajectories"""
    advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, args.tau, device)

    """perform TRPO update"""
    a2c_step(policy_net, value_net, optimizer_policy, optimizer_value, states, actions, returns, advantages, args.l2_reg)


def main_loop():
    
    hist_reward = []
    hist_maxheight = []
    hist_pos = []

    for i_iter in range(args.max_iter_num):
        """generate multiple trajectories that reach the minimum batch_size"""
        batch, log = agent.collect_samples(args.min_batch_size, render=args.render)
        t0 = time.time()
        update_params(batch)
        t1 = time.time()

        hist_reward.append({'avg': log.get('avg_reward'),
                            'min': log.get('min_reward'),
                            'max': log.get('max_reward'),
                            })
        hist_maxheight.append(log.get('max_height'))
        hist_pos.append(log.get('hist_pos'))

        """evaluate with determinstic action (remove noise for exploration)"""
        # _, log_eval = agent.collect_samples(args.eval_batch_size, mean_action=True)
        # t2 = time.time()

        if i_iter % args.log_interval == 0:
            pd.DataFrame.from_records(hist_reward).to_csv(DATA_PATH + 'reward.csv', mode='a')
            pd.Series(hist_maxheight).to_csv(DATA_PATH + 'height.csv', mode='a')
            pd.Series(hist_pos).to_csv(DATA_PATH + 'pose.csv', mode='a')
            hist_reward = []
            hist_maxheight = []
            hist_pos = []
            # print('{}\tT_sample {:.4f}\tT_update {:.4f}\tT_eval {:.4f}\ttrain_R_min {:.2f}\ttrain_R_max {:.2f}\ttrain_R_avg {:.2f}\teval_R_avg {:.2f}'.format(
            #     i_iter, log['sample_time'], t1-t0, t2-t1, log['min_reward'], log['max_reward'], log['avg_reward'], log_eval['avg_reward']))

        if args.save_model_interval > 0 and (i_iter+1) % args.save_model_interval == 0:
            breakpoint()
            to_device(torch.device('cpu'), policy_net, value_net)
            pickle.dump((policy_net, value_net, running_state),
                        open(os.path.join(DATA_PATH, 'learned_models/{}_a2c{}.p'.format("Assembly" , i_iter)), 'wb'))
            to_device(device, policy_net, value_net)

        """clean up gpu memory"""
        torch.cuda.empty_cache()

if __name__=="__main__":
    main_loop()
