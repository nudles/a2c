import argparse
import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--algo', default='a2c',
                        help='algorithm to use: a2c | ppo ')
    parser.add_argument('--lr', type=float, default=7e-5,
                        help='learning rate (default: 7e-4)')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='max norm off gradients (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--num-processes', type=int, default=1,
                        help='how many training CPU processes to use (default: 16)')
    parser.add_argument('--num-steps', type=int, default=32,
                        help='number of forward steps in A2C (default: 5)')
    parser.add_argument('--clip-param', type=float, default=0.2,
                        help='clip parameter (default: 0.2)')
    parser.add_argument('--log-interval', type=int, default=50,
                        help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--num-frames', type=int, default=80000,
                        help='number of frames to train (default: 10e6)')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--obs_size', type=int, default=200,
                        help='observation vector size')
    parser.add_argument('--cycle_len', type=int, default=500,
                        help='observation vector size')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='whether to record the logfile')
    parser.add_argument('--num_models', type=int, default=3,
                        help='number of the model to use')
    parser.add_argument('--beta', type=float, default=1,
                        help='balance the accuracy and latency when calculate the reward')
    parser.add_argument('--tau', type=float, default=2,
                        help='max waiting time for enqueue')
    parser.add_argument('--max_latency', type=float, default=16,
                        help='accept latency for each request')
    parser.add_argument('--policy', choices=['async', 'sync'], default='async', help='policy')

    args = parser.parse_args()

    print("cuda: %s" % str(args.cuda))
    if args.cuda:
        assert torch.cuda.is_available(), 'CUDA is not available in this machine!'

    return args

if __name__ == '__main__':
    get_args()