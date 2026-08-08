import time
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.nn as nn
from datetime import datetime
import torch
import random
import numpy as np
import torch.nn.functional as F
from tools import load_model

EPS = np.finfo(float).eps

# frame rates used for enrollment-cue slicing (see _run_one_epoch)
lip_FPS = 25
gesture_FPS = 15

        
class Solver(object):
    def __init__(self, train_data, validation_data, model, optimizer, args):
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
        self.model = model
        self.optimizer = optimizer
        self.lip_setting = bool(int(args.lip_setting))
        self.enroll_second = float(args.enroll_second)

        if self.args.distributed:
            self.model = DDP(self.model, find_unused_parameters=True)
      
        self._reset()


    def _reset(self):
        """Initialize training state, or resume from ``args.continue_from``."""
        self.halving = False
        self.val_no_impv = 0

        if not self.args.continue_from or self.args.continue_from.lower() == "false":
            self.best_val_loss = float("inf")
            self.start_epoch = 1
            if self.print:
                print("Start new training")
            return

        # ---------------- resume: load weights ----------------
        self.model = load_model(self.model, self.args.continue_from)

        # freeze rules: the RoBERTa transcript encoder stays a frozen feature
        # extractor; everything else remains trainable
        for name, param in self.model.named_parameters():
            if "transcript_encoder" in name:
                param.requires_grad = False
            print("ft: " if param.requires_grad else "freeze: ", name)

        # checkpoint keys: epoch / step / model_state_dict / optim / cv_log /
        # scheduler / new_bob / loss
        checkpoint = torch.load(
            self.args.continue_from, map_location="cpu", weights_only=False
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.start_epoch = checkpoint["epoch"]
        self.best_val_loss = 99999

        if self.print:
            print("Resume training from epoch: {}".format(self.start_epoch))



    def train(self):
        for epoch in range(self.start_epoch, self.args.epochs + 1):
            #     break
            self.joint_loss_weight = epoch
            if self.args.distributed:
                self.args.train_sampler.set_epoch(epoch)

            # # Train
            self.model.train()
            start = time.time()
            tr_loss = self._run_one_epoch(
                data_loader=self.train_data,
                state="train",
                epoch=epoch,
            )
            reduced_tr_loss = tr_loss


            if self.print:
                print(
                    "Train Summary | End of Epoch {0} | Time {1:.2f}s | Current time {2} |"
                    "Train Loss {3:.3f}| ".format(
                        epoch, time.time() - start, datetime.now(), reduced_tr_loss,
                    )
                )

            # Validation
            self.model.eval()
            start = time.time()
            with torch.no_grad():
                val_loss = self._run_one_epoch(
                    data_loader=self.validation_data, state="val", epoch=epoch
                )
                reduced_val_loss = val_loss

                if self.print:
                    print(
                        "Valid Summary | End of Epoch {0} | Time {1:.2f}s | Current time {2} |"
                        "Valid Loss {3:.3f}| ".format(
                            epoch,
                            time.time() - start,
                            datetime.now(),
                            reduced_val_loss,
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
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "best_val_loss": self.best_val_loss,
                    "val_no_impv": self.val_no_impv,
                }
                torch.save(checkpoint, self.args.log_name + "/model_dict_last.pt")
                if find_best_model:
                    torch.save(checkpoint, self.args.log_name + "/model_dict_best.pt")
                    print("Found new best model, dict saved")
 
                # torch.save(
                #         checkpoint,
                if epoch % 5 == 0:
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
        total_loss_speech_ce = 0
        total_loss_visual_ce = 0
        total_loss_speech_regree = 0
        total_loss_speech_ssl_regree = 0
        total_loss_mel_l1 = 0
        total_loss_av_front_align_mse_loss = 0
        total_loss_av_info_nce_loss = 0
  

        self.accu_count = 0
        self.optimizer.zero_grad()

        lip_setting = self.lip_setting
        for idx, (mixtures,enrolls,codecs,visual_codecs, vsr_sync_feat,switch_times) in enumerate(data_loader):  
         
             
            visual_sync = vsr_sync_feat.cuda().squeeze(0).float()
            tgts = mixtures
            mixtures = mixtures.cuda().squeeze(0).float()
            tgts = tgts.cuda().squeeze(0).float()
            batch_size =  int(mixtures.shape[0])
            mixtures_lengths = torch.tensor(mixtures.shape[1]).repeat(batch_size).cuda().long()
            enrolls = enrolls.cuda().squeeze(0).float()
            enrolls_lengths =  torch.tensor(enrolls.shape[1]).repeat(batch_size).cuda().long()
            codecs = codecs.cuda().squeeze(0).long()
            codecs_lengths =  torch.tensor(codecs.shape[1]).repeat(batch_size).cuda().long()
            visual_codecs = visual_codecs.cuda().squeeze(0).long()
            enroll_second = self.enroll_second

            if lip_setting:
                visual_aux = visual_sync[:,:int(lip_FPS*enroll_second),:]
            
            else:
                visual_aux = visual_sync[:,:int(gesture_FPS*enroll_second),:]
                 
                 
            switch_times =True
            if switch_times:
                 _data_res = {'text': mixtures, 'text_lengths': mixtures_lengths,'aux': enrolls, 'aux_lengths': enrolls_lengths,'codec': codecs, 'codec_lengths': codecs_lengths, 'av_token': visual_codecs, 'visual_aux':visual_aux, 'visual_sync':visual_sync, 'tgts':tgts, 'switch_times': switch_times}

            else:
                trimodal_combo = True
                if trimodal_combo:
                    _data_res = {'text': mixtures, 'text_lengths': mixtures_lengths,'aux': enrolls, 'aux_lengths': enrolls_lengths,'codec': codecs, 'codec_lengths': codecs_lengths, 'av_token': visual_codecs, 'visual_aux':visual_aux, 'visual_sync':visual_sync,'transcript_aux':transcripts_enrolls, 'tgts':tgts}

                else:
                    _data_res = {'text': mixtures, 'text_lengths': mixtures_lengths,'aux': enrolls, 'aux_lengths': enrolls_lengths,'codec': codecs, 'codec_lengths': codecs_lengths, 'av_token': visual_codecs, 'visual_aux':visual_aux, 'visual_sync':visual_sync, 'tgts':tgts}

            if state == "train":

                ## Process Mel Spectrogram ##
                loss, stats, weight = self.model(**_data_res)
                self.accu_count += 1
                step += 1
                total_loss += loss.item()
                total_loss_speech_ce += stats['nll_loss'].item()
                total_loss_visual_ce += stats['nll_visual_loss'].item()
                total_loss_speech_regree += stats['reg_loss'].item()
                
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad()
                # print(loss)
                
                #     print(
                #         "step:{}/{} avg loss:{:.3f}".format(
                #             step, total_step, total_loss / step
                if idx % 1000 == 0:
                    denom = step if step > 0 else 1
                    print(
                        f"step: {step}/{total_step} | "
                        # f"mel_l1: {total_loss_mel_l1/denom:.3f} | "
                        f"avg_ce: {total_loss/denom:.3f} | "
                        f"speech_ce: {total_loss_speech_ce/denom:.3f} | "
                        f"visual_ce: {total_loss_visual_ce/denom:.3f} | "
                        f"regre: {total_loss_speech_regree/denom:.3f} | "
                        # f"ssl mse: {total_loss_speech_ssl_regree/denom:.3f} |"
                        # f"av_front_align mse: {total_loss_av_front_align_mse_loss/denom:.3f} | "
                        # f"av_info_nce_loss : {total_loss_av_info_nce_loss/denom:.3f} | "
                    )
               
              
            else:
                loss, stats, weight = self.model(**_data_res)
                total_loss += loss.item()

                total_loss_speech_ce += stats['nll_loss'].item()
                total_loss_visual_ce += stats['nll_visual_loss'].item()
                total_loss_speech_regree += stats['reg_loss'].item()
                step += 1

                if idx % 1000 == 0:
                    denom = step if step > 0 else 1
                    print(
                        f"step: {step}/{total_step} | "
                        # f"mel_l1: {total_loss_mel_l1/denom:.3f} | "
                        f"avg_ce: {total_loss/denom:.3f} | "
                        f"speech_ce: {total_loss_speech_ce/denom:.3f} | "
                        f"visual_ce: {total_loss_visual_ce/denom:.3f} | "
                        f"regre: {total_loss_speech_regree/denom:.3f} | "
                        # f"ssl mse: {total_loss_speech_ssl_regree/denom:.3f} | "
                        # f"av_front_align mse: {total_loss_av_front_align_mse_loss/denom:.3f} | "
                        # f"av_info_nce_loss : {total_loss_av_info_nce_loss/denom:.3f} | "
                         
                    )
                 
                #     print(
                #         "step:{}/{} avg loss:{:.3f}".format(
                #             step, total_step, total_loss / step
                # pass
                
        return total_loss / (idx + 1)

 
    def _reduce_tensor(self, tensor):
        if not self.args.distributed:
            return tensor
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.args.world_size
        return rt
