import argparse
import torch
from dataload import get_dataloader
import os
from solver import Solver

import torch.nn as nn
import pdb
import time
from _funcodec import build_model
import yaml
from funcodec.torch_utils.load_pretrained_model import load_pretrained_model


def main(args):
    if args.distributed:
        # dist configs
        torch.manual_seed(3407)
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    #=====================================for larual gpt============================================
 
    ## load laura gpt model
    model: nn.Module = build_model(args)
    # model.cuda()
    # l.info(f"model {model} is intialized")
    # l.info(f"model parameters: {sum(p.numel() for p in model.parameters())}")
    # l.info(f"auto-regressive decoder-only LM parameters: {sum(p.numel() for p in model.codec_lm.parameters())}")
    for p in args.init_param:
        print(f"Loading pretrained params from {p}")
        load_pretrained_model(
            model=model,
            init_param=p,
            ignore_init_mismatch=True,
            # NOTE(kamo): "cuda" for torch.load always indicates cuda:0
            #   in PyTorch<=1.4
            map_location=f"cuda:{torch.cuda.current_device()}",
        )
    #=====================================for larual gpt============================================

    if (args.distributed and args.local_rank == 0) or args.distributed == False:
        print("started on " + args.log_name + "\n")
        print(args)
        print(
            "\nTotal number of parameters: {} \n".format(
                sum(p.numel() for p in model.parameters())
            )
        )
        print(model)
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_sampler, train_generator = get_dataloader(args, partition="train")
    # train_sampler, train_generator = get_dataloader(args, partition="test")
    _, val_generator = get_dataloader(args, partition="test")
    args.train_sampler = train_sampler
    solver = Solver(
        args=args,
        model=model,
        optimizer=optimizer,
        train_data=train_generator,
        validation_data=val_generator,
    )
    solver.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("online")

      # Dataloader
    parser.add_argument(
        "--mix_lst_path",
        type=str,
        default="",
        help="directory including train data",
    )
    parser.add_argument(
        "--codec_lst_path",
        type=str,
        default="",
        help="directory including train data",
    )

     
    parser.add_argument(
        "--mix_lst_train_path",
        type=str,
        default="",
        help="directory including train data",
    )
    parser.add_argument(
        "--codec_lst_train_path",
        type=str,
        default="",
        help="directory including train data",
    )
    parser.add_argument(
        "--vsr_sync_lst_train_path",
        type=str,
        default="",
        help="directory including train data",
    )
    parser.add_argument(
        "--vsr_sync_lst_path",
        type=str,
        default="",
        help="directory including train data",
    )
     

    parser.add_argument(
        "--aux_train_list",
        type=str,
        default="",
        help="directory including train data",
    )
    parser.add_argument(
        "--aux_list",
        type=str,
        default="",
        help="directory including train data",
    )
    parser.add_argument(
        "--visual_codec_lst_train_path",
        type=str,
        default="",
        help="directory including train data",
    )
     
     
    parser.add_argument(
        "--visual_dir",
        type=str,
        default="",
        help="directory including train data",
    )
    parser.add_argument(
        "--vsr_feat_dir",
        type=str,
        default="",
        help="directory including train data",
    )
     
    
    # Training
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size")
     
    parser.add_argument(
        "--max_length", default=6, type=int, help="max_length of mixture in training",
    )
    parser.add_argument(
        "--max_enroll_length", default=2, type=int, help="max_length of mixture in training",
    )
     
    parser.add_argument(
        "--num_workers",
        default=4,
        type=int,
        help="Number of workers to generate minibatch",
    )
    parser.add_argument(
        "--epochs", default=100, type=int, help="Number of maximum epochs"
    )
    parser.add_argument(
        "--effec_batch_size", default=8, type=int, help="effective Batch size"
    )
    parser.add_argument(
        "--accu_grad", default=0, type=int, help="whether to accumulate grad"
    )
     # optimizer
    parser.add_argument("--lr", default=1e-3, type=float, help="Init learning rate")
    parser.add_argument(
        "--max_norm", default=5, type=float, help="Gradient norm threshold to clip",
    )

    # Log and Visulization
    parser.add_argument(
        "--log_name", type=str, default=None, help="the name of the log"
    )
    parser.add_argument(
        "--use_tensorboard", type=int, default=0, help="Whether to use use_tensorboard",
    )
    parser.add_argument(
        "--continue_from", type=str, default="", help="Whether to use use_tensorboard",
    )
    
    parser.add_argument(
        "--mix_csv_path", type=str, default="", help="Whether to use use_tensorboard",
    )
    parser.add_argument(
        "--switch_audio_dir", type=str, default="", help="Whether to use use_tensorboard",
    )
    parser.add_argument(
        "--codec_dir", type=str, default="", help="Whether to use use_tensorboard",
    )
    parser.add_argument(
        "--use_visual_aux", type=str, default="", help="Whether to use use_tensorboard",
    )
    parser.add_argument(
        "--Laura", type=str, default="", help="Whether to use use_tensorboard",
    )
    parser.add_argument(
        "--lip_setting", type=str, default="", help="Whether to use use_tensorboard",
    )
    parser.add_argument(
        "--enroll_second", type=str, default="", help="Whether to use use_tensorboard",
    )
     

    # Distributed training
    parser.add_argument("--local-rank", default=0, type=int)
    # parameters for fbank of spekaker model
    parser.add_argument("--sample_rate", default=16000, type=int)
    parser.add_argument("--config", type=str, default=None, help="path to yaml config")
    # parser.add_argument("--win", default=512, type=int)
    # parser.add_argument("--hop_length", default=128, type=int)
    # parser.add_argument("--n_mels", default=80, type=int)
    # parser.add_argument("--log", required=True, type=str, help="Output of the log")
    # parser.add_argument("--config", type=str, default=None, help="path to yaml config")
    # parser.add_argument("--ckpt_path", type=str, required=True)
    # parser.add_argument("--resume", type=str, nargs="?", const="")
    # parser.add_argument("--fine_tune", type=str, default="") ## Determines if we finetue or not. 
    parser.add_argument("--model_name", type=str, default=None,
                        help="model name in MODEL_REGISTRY (src/model_registry.py); "
                             "overrides env LAURA_MODEL_NAME / default")
    parser.add_argument("--data_mode", type=str, default="normal",
                        help="dataset selection for dataload.get_dataloader: "
                             "normal | vocc_vox2 | vocc_lrs3 | scale | scale_trimodal | "
                             "gesture | gesture_trimodal | switch | switch_sws_token | memo")
    parser.add_argument("--text_direc", type=str, default="",
                        help="transcript directory (scale_trimodal / gesture_trimodal)")
    args = parser.parse_args()
    if args.config is not None:
        with open(args.config, "r") as file:
            config = yaml.safe_load(file)
        for k, v in config.items():
            args.__setattr__(k, v)
    args.init_param = [f"{args.codec_model_file}:quantizer.rq.model:quantizer_codebook"]
    args.distributed = False
    args.world_size = 1
    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1
        args.world_size = int(os.environ["WORLD_SIZE"])

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."
    
    main(args)
