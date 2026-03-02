import numpy as np
from tqdm import tqdm
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import AverageMeter, recall_at_k, ndcg_k
class Trainer:
    def __init__(self, model, train_dataloader, eval_dataloader, test_dataloader, device, args):

        self.args = args
        self.device = device
        self.model = model

        self.batch_size = self.args.batch_size
        self.sim=self.args.sim

        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader


        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    
    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader)

    def valid(self, epoch):
        return self.iteration(epoch, self.eval_dataloader, train=False)

    def test(self, epoch):
        return self.iteration(epoch, self.test_dataloader, train=False)

    def iteration(self, epoch, dataloader, train=True):
        raise NotImplementedError
    
    def predict_full(self, seq_out):
        test_item_emb = self.model.item_embeddings.weight
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred
    
    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HIT@5": "{:.4f}".format(recall[0]),
            "NDCG@5": "{:.4f}".format(ndcg[0]),
            "HIT@10": "{:.4f}".format(recall[1]),
            "NDCG@10": "{:.4f}".format(ndcg[1]),
            "HIT@20": "{:.4f}".format(recall[3]),
            "NDCG@20": "{:.4f}".format(ndcg[3]),
        }
        print(post_fix)
        with open(self.args.log_file, "a") as f:
            f.write(str(post_fix) + "\n")
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)
    

class VQICLTrainer(Trainer):
    def __init__(self, model, train_dataloader, eval_dataloader, test_dataloader, device, args):
        super(VQICLTrainer, self).__init__(model, train_dataloader, eval_dataloader, test_dataloader, device, args)

        self.model = model.to(self.device)
        
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        self.optimizer = self._create_optimizer()
        self.criterion = nn.CrossEntropyLoss(ignore_index=0) 
        

    def _create_optimizer(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        return optimizer

    def _create_contrastive_loss(self, h_orig, h_aug):
        h_orig = F.normalize(h_orig, p=2, dim=1)
        h_aug = F.normalize(h_aug, p=2, dim=1)
        batch_size = h_orig.size(0)
        out = torch.cat([h_orig, h_aug], dim=0)
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.args.temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
        pos_sim = torch.exp(torch.sum(h_orig * h_aug, dim=-1) / self.args.temperature)
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        return loss

    def iteration(self, epoch, dataloader, train=True):
        if train:

            self.model.train()
            rec_losses, recon_losses, vq_losses, cl_losses, total_losses = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
            
            rec_t_data_iter = tqdm(enumerate(dataloader), desc=f"Training Epoch {epoch}/{self.args.epochs}" , total=len(dataloader))

            for i, (rec_batch) in rec_t_data_iter:
                rec_batch = tuple(t.to(self.device) for t in rec_batch)
                _, input_seq, input_seq_len, target_pos, target = rec_batch
                seq_out = input_seq
                pos_ids = target_pos
                
                self.optimizer.zero_grad()
                
                outputs = self.model(seq_out)

                seq_rec_emb , recon_logits, vq_loss, seq_aug_emb = outputs
                
                logits = self.predict_full(seq_rec_emb)
                
                rec_loss = self.criterion(logits, pos_ids[:, -1])
                recon_loss = self.criterion(recon_logits.view(-1, self.args.item_size), seq_out.view(-1))
                cl_loss = self._create_contrastive_loss(seq_rec_emb, seq_aug_emb)
                
                total_loss = (self.args.rec_loss_weight * rec_loss + 
                            self.args.recon_loss_weight * recon_loss + 
                            self.args.vq_loss_weight * vq_loss + 
                            self.args.cl_loss_weight * cl_loss)
                
                total_loss.backward()
                self.optimizer.step()
                
                rec_losses.update(rec_loss.item(), seq_out.size(0))
                recon_losses.update(recon_loss.item(), seq_out.size(0))
                vq_losses.update(vq_loss.item(), seq_out.size(0))
                cl_losses.update(cl_loss.item(), seq_out.size(0))
                total_losses.update(total_loss.item(), seq_out.size(0))
                
                rec_t_data_iter.set_postfix(
                    TotalLoss=f"{total_losses.avg:.4f}", RecLoss=f"{rec_losses.avg:.4f}",
                    ReconLoss=f"{recon_losses.avg:.4f}", VQLoss=f"{vq_losses.avg:.4f}", CLLoss=f"{cl_losses.avg:.4f}"
                )


        else:
            rec_data_iter = tqdm(enumerate(dataloader), desc=f"=Validation/Testing=", total=len(dataloader)) 
            self.model.eval()                                   

            with torch.no_grad():
                for i, batch in rec_data_iter:
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, input_seq_len, target_pos, answers = batch

                    outputs = self.model(input_ids)
                    seq_rec_emb, _, _, _ = outputs

                    scores = self.predict_full(seq_rec_emb)

                    rating_pred = scores.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()

                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0

                    ind = np.argpartition(rating_pred, -20)[:, -20:] 
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1] 
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()

                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0) 

            metrices_list, log_info = self.get_full_sort_score(epoch, answer_list, pred_list)
            return metrices_list, log_info
