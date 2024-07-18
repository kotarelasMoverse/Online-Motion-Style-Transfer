import argparse
import json
import logging
import os
import torch
import time
import numpy as np
import sys
import os
from statistics import mean
import glob

from model import RecurrentStylization, ContentClassification
from torch.utils.tensorboard import SummaryWriter
from dataloader import MotionDataset
from postprocess import save_bvh_from_network_output, remove_fs
from utils.animation_data import AnimationData

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

content_labels = ["walk", "run", "jump", "punch", "kick"]
content_labels_ids = {"01":"walk", "13":"run", "16":"jump", "18":"punch", "22":"kick"}
style_labels = ["angry", "childlike", "depressed", "old", "proud", "sexy", "strutting"]

class ArgParserTest(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # setup hyperparameters
        self.add_argument('--content_num', type=int, default=5)
        self.add_argument('--style_num', type=int, default=7)
        self.add_argument('--episode_length', type=int, default=24)
        self.add_argument("--data_path", default='./data/xia_test')
        self.add_argument("--work_dir", default='./experiments/')
        self.add_argument("--load_dir", default=None, type=str)
        self.add_argument("--input_motion", default=None, type=str)
        self.add_argument("--input_contents", default="all", nargs='+', type=str, help='The contents to take into account.')
        self.add_argument("--input_styles", default="all", nargs='+', type=str, help='The input styles to take into account.')
        self.add_argument("--target_styles", default="all", nargs='+', type=str, help='The output styles to create.')
        self.add_argument("--epochs_to_test", default=0, nargs='+', type=int, help='The model checkpoints to use.') 

        # model hyperparameters
        self.add_argument("--no_pos", default=True, action='store_true')
        self.add_argument("--no_vel", default=False, action='store_true')
        self.add_argument("--encoder_layer_num", type=int, default=2)
        self.add_argument("--decoder_layer_num", type=int, default=4)
        self.add_argument("--discriminator_layer_num", type=int, default=4)
        self.add_argument("--classifier_layer_num", type=int, default=5)
        self.add_argument("--latent_dim", type=int, default=32)
        self.add_argument("--neutral_layer_num", type=int, default=4)
        self.add_argument("--style_layer_num", type=int, default=6)
        self.add_argument("--feature_dim", type=int, default=16)

def process_single_bvh(filename, input_content, input_style, target_style, downsample=4, skel=None):

    anim = AnimationData.from_BVH(filename, downsample=downsample, skel=skel, trim_scale=4)

    episode_length = anim.len
    style_array = np.zeros(len(style_labels))
    if input_style != "neutral":
        style_array[style_labels.index(input_style)] = 1.0

    target_style_list = np.zeros(len(style_labels))
    if target_style != "neutral":
        target_style_list[style_labels.index(target_style)] = 1.0
    content_index = content_labels.index(input_content)
    data = {
            "rotation": torch.FloatTensor(anim.get_joint_rotation()).to(device),
            "position": torch.FloatTensor(anim.get_joint_position()).to(device),
            "velocity": torch.FloatTensor(anim.get_joint_velocity()).to(device),
            "content": torch.FloatTensor(np.tile([np.eye(len(content_labels))[content_index]],(episode_length, 1))).to(device),
            "contact": torch.FloatTensor(anim.get_foot_contact(transpose=False)).to(device),
            "root": torch.FloatTensor(anim.get_root_posrot()).to(device),
            "input_style": torch.FloatTensor(np.tile([style_array],(episode_length,1))).to(device),
            "transferred_style": torch.FloatTensor(np.tile([target_style_list],(episode_length,1))).to(device),
            "content_index": torch.LongTensor(content_index).to(device)
        }
    
    dim_dict = {
            "rotation": data["rotation"][0].shape[-1],
            "position": data["position"][0].shape[-1],
            "velocity": data["velocity"][0].shape[-1],
            "style": len(style_labels), 
            "content": len(content_labels),
            "contact": data["contact"][0].shape[-1],
            "root": data["root"][0].shape[-1]
        }

    # rotation torch.Size([80, 124])
    # position torch.Size([80, 60])
    # velocity torch.Size([80, 60])
    # content torch.Size([80, 5])
    # contact torch.Size([80, 4])
    # root torch.Size([4, 80])
    # input_style torch.Size([80, 7])
    # transferred_style torch.Size([80, 7])
    # content_index torch.Size([0])
    return (
        data, # dict_keys(['rotation', 'position', 'velocity', 'content', 'contact', 'root', 'input_style', 'transferred_style', 'content_index'])
        dim_dict # {'rotation': 124, 'position': 60, 'velocity': 60, 'style': 7, 'content': 5, 'contact': 4, 'root': 80}
    )

def main():
    args = ArgParserTest().parse_args()
    # # Custom argument override
    # args.data_path = r'.\data\xia_test'
    # args.load_dir = r'C:\Users\info\Documents\GitHub\gists\models\style_erd\experiments\07-08-11-39_seed_88_train_Style_ERD'
    # args.input_contents = ['all']
    # args.input_styles = ['all']
    # args.target_styles = ['all']
    # args.epochs_to_test = [2000]
    # args.no_pos = True
    # #
    print(args.input_contents)
    print(args.input_styles)
    print(args.target_styles)
    print(args.epochs_to_test)
    assert (set(args.input_contents).issubset(set(content_labels)) or args.input_contents[0] == 'all')
    assert (set(args.input_styles).issubset(set(style_labels)) or args.input_styles[0] == 'all')
    assert (set(args.target_styles).issubset(set(style_labels)) or args.target_styles[0] == 'all')

    if args.input_contents[0] == 'all': args.input_contents = content_labels
    if args.input_styles[0] == 'all': args.input_styles = style_labels + ['neutral']
    if args.target_styles[0] == 'all': args.target_styles = style_labels + ['neutral']

    # Setup output folders
    for epoch in args.epochs_to_test:
        output_folder = os.path.join(args.load_dir, 'stylized_xia_test', str(epoch))
        os.makedirs(output_folder, exist_ok=True)

    # Retrieve input .bvh files
    all_bvh_list = [f for f in os.listdir(args.data_path) if f.endswith('.bvh')]
    # Filter .bvh files
    input_bvh_list = []
    for f in all_bvh_list:
        tags = f.split('_')
        if (tags[0] in args.input_styles and tags[1] in list(content_labels_ids.keys()) and content_labels_ids[tags[1]] in args.input_contents):
            input_bvh_list.append(f)

    for epoch in args.epochs_to_test:
        output_folder = os.path.join(args.load_dir, 'stylized_xia_test', str(epoch))

        for input_file in input_bvh_list:
            tags = input_file.split('_')
            input_style = tags[0]
            input_content = content_labels_ids[tags[1]]

            for target_style in args.target_styles:
                test_data, dim_dict = process_single_bvh(os.path.join(args.data_path, input_file), input_content, input_style, target_style, downsample=2)
                model = RecurrentStylization(args, dim_dict).to(device)
                model.load_state_dict(torch.load(os.path.join(args.load_dir, 'model', 'model_{}.pt'.format(epoch))))
                model.eval()
                for (key,val) in test_data.items():
                    if key != "content_index":
                        test_data[key] = val.unsqueeze(0)
                
                transferred_motion = model.forward_gen(test_data["rotation"], test_data["position"], test_data["velocity"],
                                                       test_data["content"], test_data["contact"],
                                                       test_data["input_style"], test_data["transferred_style"], test_time=True)
                
                transferred_motion = transferred_motion["rotation"].squeeze(0)
                root_info = test_data["root"].squeeze(0).transpose(0, 1)
                foot_contact = test_data["contact"].cpu().squeeze(0).transpose(0, 1).numpy()
                transferred_motion = torch.cat((transferred_motion,root_info), dim=-1).transpose(0, 1).detach().cpu()
                remove_fs(
                    transferred_motion,
                    foot_contact,
                    output_path=os.path.join(output_folder, '{}_{}_to_{}_{}.bvh'.format(epoch, input_style, target_style, input_content))
                )

    # test_data, dim_dict = process_single_bvh(args.input_motion, args, downsample=2)

    # model = RecurrentStylization(args, dim_dict).to(device)

    # model.load_state_dict(torch.load(os.path.join(args.load_dir, 'model/model_1050.pt')))
    # model.eval()

    # for (key,val) in test_data.items():
    #     if key != "content_index":
    #         test_data[key] = val.unsqueeze(0)
    

    # content = args.input_content
    # input_style = args.input_style
    # output_style = args.target_style

    # transferred_motion = model.forward_gen(test_data["rotation"], test_data["position"], test_data["velocity"],
    #                     test_data["content"], test_data["contact"],
    #                     test_data["input_style"], test_data["transferred_style"], test_time=True)
    
    # transferred_motion = transferred_motion["rotation"].squeeze(0)
    # root_info = test_data["root"].squeeze(0).transpose(0, 1)
    # foot_contact = test_data["contact"].cpu().squeeze(0).transpose(0, 1).numpy()
    # transferred_motion = torch.cat((transferred_motion,root_info), dim=-1).transpose(0, 1).detach().cpu()
    # remove_fs(
    #     transferred_motion,
    #     foot_contact,
    #     output_path="output_dir/{}_to_{}_{}.bvh".format(input_style, output_style, content)
    # )


if __name__ == "__main__":
    main()