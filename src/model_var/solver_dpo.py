import time
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from datetime import datetime
import torch
import numpy as np
from tools import load_model
from dpo_loss_githubv2 import DPOLoss

EPS = np.finfo(float).eps
        
class Solver(object):
    def __init__(self, train_data, validation_data, ref_model, policy_model, optimizer, args):
        self.train_data = train_data
        self.validation_data = validation_data
        self.args = args
        self.grad_clip = args.grad_clip
        self.print = False
        if (
            self.args.distributed and self.args.local_rank == 0
        ) or not self.args.distributed:
            self.print = True
            if self.args.use_tensorboard:
                self.writer = SummaryWriter("%s/tensorboard/" % args.log_name)
        self.ref_model = ref_model
        self.policy_model = policy_model
        self.dpo_loss_func = DPOLoss()
        self.optimizer = optimizer
         

        if self.args.distributed:
            self.ref_model = DDP(self.ref_model, find_unused_parameters=True)
            self.policy_model = DDP(self.policy_model, find_unused_parameters=True)
      
        self._reset()


    def _reset(self):
        """Initialize DPO training state, or resume from ``args.continue_from``."""
        self.halving = False
        self.val_no_impv = 0

        if not self.args.continue_from or self.args.continue_from.lower() == "false":
            self.best_val_loss = float("inf")
            self.start_epoch = 1
            if self.print:
                print("Start new training")
            return

        # ---------------- resume: load weights ----------------
        # reference model: frozen copy of the SFT checkpoint
        self.ref_model = load_model(self.ref_model, self.args.continue_from)
        # policy model: the trainable copy
        self.policy_model = load_model(self.policy_model, self.args.continue_from)

        for param in self.ref_model.parameters():
            param.requires_grad = False
        for param in self.policy_model.parameters():
            param.requires_grad = True

        for name, param in self.ref_model.named_parameters():
            print(f"{name}: requires_grad={param.requires_grad}")
        for name, param in self.policy_model.named_parameters():
            print(f"{name}: requires_grad={param.requires_grad}")

        # checkpoint keys: epoch / step / model_state_dict / optim / cv_log /
        # scheduler / new_bob / loss
        checkpoint = torch.load(
            self.args.continue_from, map_location="cpu", weights_only=False
        )
        self.optimizer.load_state_dict(checkpoint["optim"])
        self.start_epoch = checkpoint["epoch"]
        self.best_val_loss = 99999

        if self.print:
            print("Resume training from epoch: {}".format(self.start_epoch))


    def train(self):
        for epoch in range(self.start_epoch, self.args.epochs + 1):
            self.joint_loss_weight = epoch
            if self.args.distributed:
                self.args.train_sampler.set_epoch(epoch)

            # # Train
            self.policy_model.train()
            start = time.time()
            tr_loss, tr_loss_policy, tr_loss_ref  = self._run_one_epoch(
                data_loader=self.train_data,
                state="train",
                epoch=epoch,
            )
            reduced_tr_loss = self._reduce_tensor(tr_loss)

            if self.print:
                print(
                    "Train Summary | End of Epoch {0} | Time {1:.2f}s | Current time {2} |"
                    "Train Loss {3:.3f}|Train Loss policy {3:.3f}|Train Loss ref {3:.3f}| ".format(
                        epoch, time.time() - start, datetime.now(), reduced_tr_loss,tr_loss_policy, tr_loss_ref
                    )
                )

            # Validation
            self.policy_model.eval()
            start = time.time()
            with torch.no_grad():
                val_loss, val_loss_policy, val_loss_ref  = self._run_one_epoch(
                    data_loader=self.validation_data, state="val", epoch=epoch
                )
                reduced_val_loss = self._reduce_tensor(val_loss)
                if self.print:
                    print(
                        "Valid Summary | End of Epoch {0} | Time {1:.2f}s | Current time {2} |"
                        "Valid Loss {3:.3f}| valid loss policy {3:.3f}| valid loss ref {3:.3f}| ".format(
                            epoch,
                            time.time() - start,
                            datetime.now(),
                            reduced_val_loss,
                            val_loss_policy, 
                            val_loss_ref
                        )
                    )

            # Check whether to adjust learning rate and early stop
            find_best_model = False
            if reduced_val_loss >= self.best_val_loss:
                self.val_no_impv += 1
                if self.val_no_impv >= 10:
                    if self.print:
                        print("No improvement for 10 epochs, early stopping.")
                    break
            else:
                self.val_no_impv = 0
                self.best_val_loss = reduced_val_loss
                find_best_model = True

            if self.val_no_impv == 6:
                self.halving = True

            # Halving the learning rate
            if self.halving:
                optim_state = self.optimizer.state_dict()
                optim_state["param_groups"][0]["lr"] = (
                    optim_state["param_groups"][0]["lr"] / 2
                )
                self.optimizer.load_state_dict(optim_state)
                if self.print:
                    print(
                        "Learning rate adjusted to: {lr:.6f}".format(
                            lr=optim_state["param_groups"][0]["lr"]
                        )
                    )
                self.halving = False

            if self.print:
                # Tensorboard logging
                if self.args.use_tensorboard:
                    self.writer.add_scalar("Train_loss", reduced_tr_loss, epoch)
                    self.writer.add_scalar("Validation_loss", reduced_val_loss, epoch)

                # Save model
                checkpoint = {
                    "model": self.policy_model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "best_val_loss": self.best_val_loss,
                    "val_no_impv": self.val_no_impv,
                }
                torch.save(
                        checkpoint,
                        self.args.log_name + "/model_dict_" + str(epoch) + ".pt",
                    )

    def _run_one_epoch(
        self, data_loader, state, epoch
    ):
        step = 0
        total_step = len(data_loader)
        total_loss = 0
        total_loss_policy = 0
        total_loss_ref = 0

        self.accu_count = 0
        self.optimizer.zero_grad()
        for idx, (mixtures,enrolls,codecs, codecs_rej) in enumerate(data_loader):
            mixtures = mixtures.cuda().squeeze(0).float()
            batch_size =  int(mixtures.shape[0])
            mixtures_lengths = torch.tensor(mixtures.shape[1]).repeat(batch_size).cuda().long()
            enrolls = enrolls.cuda().squeeze(0).float()
            enrolls_lengths =  torch.tensor(enrolls.shape[1]).repeat(batch_size).cuda().long()
            codecs = codecs.cuda().squeeze(0).long()
            codecs_lengths =  torch.tensor(codecs.shape[1]).repeat(batch_size).cuda().long()
            codecs_rej = codecs_rej.cuda().squeeze(0).long()
            codecs_rej_lengths =  torch.tensor(codecs_rej.shape[1]).repeat(batch_size).cuda().long()
            _data_res = {'text': mixtures, 'text_lengths': mixtures_lengths,'aux': enrolls, 'aux_lengths': enrolls_lengths,'codec': codecs, 'codec_lengths': codecs_lengths, 'codec_rej': codecs_rej, 'codec_rej_lengths': codecs_rej_lengths}

            if state == "train":
                with torch.no_grad():
                    loss_ref, stats_ref, weight_ref, reference_chosen_logps, reference_rejected_logps = self.ref_model(**_data_res)
                loss_policy, stats_policy, weight_policy,policy_chosen_logps, policy_rejected_logps = self.policy_model(**_data_res)
                loss_dpo, chosen_reward, rejected_reward = self.dpo_loss_func.compute_loss(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps)
                print(f"loss_dpo: {loss_dpo.item()}, chosen_reward: {chosen_reward.item()}, rejected_reward: {rejected_reward.item()}" )
                self.accu_count += 1
                step += 1
                total_loss += loss_policy.item() + loss_dpo.item() 
                total_loss_policy += loss_policy.item()
                total_loss_ref += loss_ref.item()

                loss_final = 0.1 * loss_policy + loss_dpo 


                loss_final.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad()

                if idx % 200 == 0:
                    print("step:{}/{} avg loss:{:.3f}, avg policy loss:{:.3f}, avg ref loss:{:.3f}".format(
                            step, total_step, total_loss / step, total_loss_policy / step, total_loss_ref / step
                        )
                    )
               
              
            else:
                with torch.no_grad():
                    loss_ref, stats_ref, weight_ref, reference_chosen_logps, reference_rejected_logps = self.ref_model(**_data_res)
                    loss_policy, stats_policy, weight_policy,policy_chosen_logps, policy_rejected_logps = self.policy_model(**_data_res)
                loss_dpo, chosen_reward, rejected_reward = self.dpo_loss_func.compute_loss(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps)
                print(f"loss_dpo: {loss_dpo.item()}, chosen_reward: {chosen_reward.item()}, rejected_reward: {rejected_reward.item()}" )
                total_loss += loss_policy.item() + loss_dpo.item() 
                total_loss_policy += loss_policy.item()
                total_loss_ref += loss_ref.item()
                step += 1
                if idx % 1000 == 0:
                    print("step:{}/{} avg loss:{:.3f}, avg policy loss:{:.3f}, avg ref loss:{:.3f}".format(
                            step, total_step, total_loss / step, total_loss_policy / step, total_loss_ref / step
                        )
                    )

        return total_loss / (idx + 1), total_loss_policy / (idx + 1), total_loss_ref / (idx + 1)

    def _reduce_tensor(self, tensor):
        if not self.args.distributed:
            return tensor
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.args.world_size
        return rt
