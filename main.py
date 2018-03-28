import glob
import os
import time
import pathlib
import logging
import datetime
#import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from arguments import get_args
# from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
# from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
# from baselines.common.vec_env.vec_normalize import VecNormalize
# from envs import make_env
from kfac import KFACOptimizer
from model import CNNPolicy, MLPPolicy
from storage import RolloutStorage
from visualize import visdom_plot
import rafiki

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

if not args.debug:
    pathlib.Path('log').mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler('log/server-%s' %
                             datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
logger.info(args)

assert args.algo in ['a2c', 'ppo', 'acktr']

num_updates = int(args.num_frames) // args.num_steps // args.num_processes

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

try:
    os.makedirs(args.log_dir)
except OSError:
    files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)


def main():
    print("#######")
    print("WARNING: All rewards are clipped or normalized so you need to use a monitor (see envs.py) or visdom plot to get true rewards")
    print("#######")

    os.environ['OMP_NUM_THREADS'] = '1'

    envs = rafiki.Envs(args.num_processes, args.num_models, args.policy,
                       args.obs_size, args.cycle_len)
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
    elif args.algo == 'acktr':
        optimizer = KFACOptimizer(actor_critic)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              obs_shape, envs.action_space, actor_critic.state_size)
    current_obs = torch.zeros(args.num_processes, *obs_shape)

    def update_current_obs(obs):
        shape_dim0 = envs.observation_space.shape[0]
        obs = torch.from_numpy(obs).float()
        current_obs[:, -shape_dim0:] = obs

    obs = envs.reset()
    update_current_obs(obs)
    rollouts.observations[0].copy_(current_obs)

    # These variables are used to compute average rewards for all processes.
    # final_rewards = torch.zeros([args.num_processes, 1])

    if args.cuda:
        current_obs = current_obs.cuda()
        rollouts.cuda()

    start = time.time()
    for j in range(num_updates):
        for step in range(args.num_steps):
            # Sample actions
            value, action, action_log_prob, states = actor_critic.act(Variable(rollouts.observations[step], volatile=True),
                                                                      Variable(rollouts.states[
                                                                               step], volatile=True),
                                                                      Variable(rollouts.masks[step], volatile=True))
            cpu_actions = action.data.squeeze(1).cpu().numpy()
            # Obser reward and next obs
            obs, reward, done, info = envs.step(cpu_actions)
            reward = torch.from_numpy(
                np.expand_dims(np.stack(reward), 1)).float()
            # final_rewards += rewards
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            if args.cuda:
                masks = masks.cuda()
            update_current_obs(obs)
            rollouts.insert(step, current_obs, states.data, action.data,
                            action_log_prob.data, value.data, reward, masks)

        next_value = actor_critic(Variable(rollouts.observations[-1], volatile=True),
                                  Variable(rollouts.states[-1], volatile=True),
                                  Variable(rollouts.masks[-1], volatile=True))[0].data

        rollouts.compute_returns(
            next_value, args.use_gae, args.gamma, args.tau)

        if args.algo in ['a2c', 'acktr']:
            values, action_log_probs, dist_entropy, states = actor_critic.evaluate_actions(Variable(rollouts.observations[:-1].view(-1, *obs_shape)),
                                                                                           Variable(rollouts.states[
                                                                                                    0].view(-1, actor_critic.state_size)),
                                                                                           Variable(rollouts.masks[
                                                                                                    :-1].view(-1, 1)),
                                                                                           Variable(rollouts.actions.view(-1, action_shape)))

            values = values.view(args.num_steps, args.num_processes, 1)
            action_log_probs = action_log_probs.view(
                args.num_steps, args.num_processes, 1)

            advantages = Variable(rollouts.returns[:-1]) - values
            value_loss = advantages.pow(2).mean()

            action_loss = -(Variable(advantages.data)
                            * action_log_probs).mean()

            if args.algo == 'acktr' and optimizer.steps % optimizer.Ts == 0:
                # Sampled fisher, see Martens 2014
                actor_critic.zero_grad()
                pg_fisher_loss = -action_log_probs.mean()

                value_noise = Variable(torch.randn(values.size()))
                if args.cuda:
                    value_noise = value_noise.cuda()

                sample_values = values + value_noise
                vf_fisher_loss = - \
                    (values - Variable(sample_values.data)).pow(2).mean()

                fisher_loss = pg_fisher_loss + vf_fisher_loss
                optimizer.acc_stats = True
                fisher_loss.backward(retain_graph=True)
                optimizer.acc_stats = False

            optimizer.zero_grad()
            (value_loss * args.value_loss_coef + action_loss -
             dist_entropy * args.entropy_coef).backward()

            if args.algo == 'a2c':
                nn.utils.clip_grad_norm(
                    actor_critic.parameters(), args.max_grad_norm)

            optimizer.step()
        elif args.algo == 'ppo':
            advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
            advantages = (advantages - advantages.mean()) / \
                (advantages.std() + 1e-5)

            for e in range(args.ppo_epoch):
                if args.recurrent_policy:
                    data_generator = rollouts.recurrent_generator(advantages,
                                                                  args.num_mini_batch)
                else:
                    data_generator = rollouts.feed_forward_generator(advantages,
                                                                     args.num_mini_batch)

                for sample in data_generator:
                    observations_batch, states_batch, actions_batch, \
                        return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample

                    # Reshape to do in a single forward pass for all steps
                    values, action_log_probs, dist_entropy, states = actor_critic.evaluate_actions(Variable(observations_batch),
                                                                                                   Variable(
                                                                                                       states_batch),
                                                                                                   Variable(
                                                                                                       masks_batch),
                                                                                                   Variable(actions_batch))

                    adv_targ = Variable(adv_targ)
                    ratio = torch.exp(action_log_probs -
                                      Variable(old_action_log_probs_batch))
                    surr1 = ratio * adv_targ
                    surr2 = torch.clamp(
                        ratio, 1.0 - args.clip_param, 1.0 + args.clip_param) * adv_targ
                    # PPO's pessimistic surrogate (L^CLIP)
                    action_loss = -torch.min(surr1, surr2).mean()

                    value_loss = (Variable(return_batch) -
                                  values).pow(2).mean()

                    optimizer.zero_grad()
                    (value_loss + action_loss - dist_entropy *
                     args.entropy_coef).backward()
                    nn.utils.clip_grad_norm(
                        actor_critic.parameters(), args.max_grad_norm)
                    optimizer.step()

        rollouts.after_update()

        if j % args.log_interval == 0:
            end = time.time()
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            print("Updates {}, num timesteps {}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".
                  format(j, total_num_steps, dist_entropy.data[0], value_loss.data[0], action_loss.data[0]))

if __name__ == "__main__":
    main()
