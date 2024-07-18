import torch
from torch.utils.tensorboard import SummaryWriter
from parse import *
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir', help='The directory of the experiment results that contain the session.log file.')
    return parser.parse_args()

def main():
    args = parse_args()
    writer = SummaryWriter(log_dir=args.experiment_dir)

    session_file_path = os.path.join(args.experiment_dir, 'session.log')

    # train_str = "Train: Epoch [1/2000], Step [60/158]| g_loss: 37.130| d_loss: 1.783| gp_loss: 0.000| r_loss: 31.023| p_loss: 11.506| v_loss: 0.072| per_loss: 2.817 | a_loss: 0.000\n"
    # test_str = "Test: Epoch [1050/2000]| g_loss: 6.718| r_loss: 6.214| p_loss: 0.812| v_loss: 0.028"

    train_fmt = "Train: Epoch [{:d}/5000], Step [{:d}/158]| g_loss: {:f}| d_loss: {:f}| gp_loss: {:f}| r_loss: {:f}| p_loss: {:f}| v_loss: {:f}| per_loss: {:f} | a_loss: {:f}\n"
    test_fmt = "Test: Epoch [{:d}/5000]| g_loss: {:f}| r_loss: {:f}| p_loss: {:f}| v_loss: {:f}"

    train_g_loss = {}
    train_d_loss = {}
    train_gp_loss = {}
    train_r_loss = {}
    train_p_loss = {}
    train_v_loss = {}
    train_per_loss = {}
    train_a_loss = {}

    test_g_loss = {}
    test_r_loss = {}
    test_p_loss = {}
    test_v_loss = {}

    with open(session_file_path, 'r') as f:
        for line in f:
            if line.split(':')[0] == 'Train':
                parsed = parse(train_fmt, line)
                train_g_loss[parsed[0]] = parsed[2]
                train_d_loss[parsed[0]] = parsed[3]
                train_gp_loss[parsed[0]] = parsed[4]
                train_r_loss[parsed[0]] = parsed[5]
                train_p_loss[parsed[0]] = parsed[6]
                train_v_loss[parsed[0]] = parsed[7]
                train_per_loss[parsed[0]] = parsed[8]
                train_a_loss[parsed[0]] = parsed[9]
            else:
                if line.split(':')[0] == 'Test':
                    parsed = parse(test_fmt, line.rstrip())
                    test_g_loss[parsed[0]] = parsed[1]
                    test_r_loss[parsed[0]] = parsed[2]
                    test_p_loss[parsed[0]] = parsed[3]
                    test_v_loss[parsed[0]] = parsed[4]

    for epoch in train_g_loss.keys():
        writer.add_scalar('Loss/Train/g_loss', train_g_loss[epoch], epoch)
        writer.add_scalar('Loss/Train/d_loss', train_d_loss[epoch], epoch)
        writer.add_scalar('Loss/Train/gp_loss', train_gp_loss[epoch], epoch)
        writer.add_scalar('Loss/Train/r_loss', train_r_loss[epoch], epoch)
        writer.add_scalar('Loss/Train/p_loss', train_p_loss[epoch], epoch)
        writer.add_scalar('Loss/Train/v_loss', train_v_loss[epoch], epoch)
        writer.add_scalar('Loss/Train/per_loss', train_per_loss[epoch], epoch)
        writer.add_scalar('Loss/Train/a_loss', train_a_loss[epoch], epoch)

    for epoch in test_g_loss.keys():
        writer.add_scalar('Loss/Test/g_loss', test_g_loss[epoch], epoch)
        writer.add_scalar('Loss/Test/r_loss', test_r_loss[epoch], epoch)
        writer.add_scalar('Loss/Test/p_loss', test_p_loss[epoch], epoch)
        writer.add_scalar('Loss/Test/v_loss', test_v_loss[epoch], epoch)

    writer.flush()

if __name__ == "__main__":
    main()