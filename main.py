import glob
import os
import pathlib
import logging
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from arguments import get_args
# from envs import make_env
from model import MLPPolicy
from storage import RolloutStorage
import rafiki
from info import Info

# create logger
logger = logging.getLogger('Rafiki')
logger.setLevel(logging.INFO)
# create console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

args = get_args()

if args.debug:
    pathlib.Path('log').mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler('log/server-%s' % datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
logger.info(args)

assert args.algo in ['a2c', 'ppo']

num_updates = int(args.num_frames) // args.num_steps // args.num_processes

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)




def main():
    print("#######")
    print("WARNING: All rewards are not clipped or normalized ")
    print("#######")

    os.environ['OMP_NUM_THREADS'] = '1'

    envs = rafiki.Envs(args.num_processes, args.num_models, args.policy,
                       args.beta, args.obs_size, args.max_latency, args.tau, args.cycle_len)
    obs_shape = envs.observation_space.shape

    actor_critic = MLPPolicy(obs_shape[0], envs.action_space)

    if envs.action_space.__class__.__name__ == "Discrete":
        action_shape = 1
    else:
        action_shape = envs.action_space.shape[0]

    if args.cuda:
        actor_critic.cuda()

    if args.algo == 'a2c':
        optimizer = optim.RMSprop(actor_critic.parameters(
        ), args.lr, eps=args.eps, alpha=args.alpha)
    elif args.algo == 'ppo':
        optimizer = optim.Adam(actor_critic.parameters(),
                               args.lr, eps=args.eps)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,obs_shape, envs.action_space)
    current_obs = torch.zeros(args.num_processes, *obs_shape)

    def update_current_obs(obs):
        shape_dim0 = envs.observation_space.shape[0]
        obs = torch.from_numpy(obs).float()
        current_obs[:, -shape_dim0:] = obs

    obs = envs.reset()
    update_current_obs(obs)
    rollouts.observations[0].copy_(current_obs)

    if args.cuda:
        current_obs = current_obs.cuda()
        rollouts.cuda()

    info_set = Info(args)

    for j in range(num_updates):
        for step in range(args.num_steps):
            logger.info('------------%d----------------' % j)
            # Sample actions
            with torch.no_grad():
                action, probs, action_log_prob = actor_critic.act(Variable(rollouts.observations[step]))
            cpu_actions = action.data.squeeze(1).cpu().numpy()
            # Obser reward and next obs
            logger.info(probs)
            obs, reward, info = envs.step(cpu_actions)
            info_set.insert(info)

            reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()

            update_current_obs(obs)
            rollouts.insert(step, current_obs, action.data, action_log_prob.data, reward)

        if args.algo in ['a2c', 'ppo']:
            action_log_probs, dist_entropy = actor_critic.evaluate_actions(Variable(rollouts.observations[:-1].view(-1, *obs_shape)),
                                                                                           Variable(rollouts.actions.view(-1, action_shape)))

            R = rollouts.rewards.detach()

            optimizer.zero_grad()
            policy_loss = - R.reshape(args.num_steps,args.num_processes).mul(action_log_probs)
            policy_loss = sum(policy_loss)/len(policy_loss)
            policy_loss.backward()

            # nn.utils.clip_grad_norm_(actor_critic.parameters(), args.max_grad_norm)

            optimizer.step()

        with torch.no_grad():
            action, probs, action_log_prob = actor_critic.act(Variable(rollouts.observations[-1]))
        logger.info(probs)

        rollouts.after_update()

        if j % args.log_interval == 0:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            print("Updates {}, num timesteps {}, reward {}, policy loss {}".
                  format(j, total_num_steps, R.data ,policy_loss.reshape(-1).data))

    logger.info(args)
    info_set.show()


if __name__ == "__main__":
    main()
