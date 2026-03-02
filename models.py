# -*- coding: utf-8 -*-
import math
import os
import pickle
import numpy as np
import random
import copy

import torch
import torch.nn.functional as F
import torch.nn as nn
from modules import *


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        
        distances = (torch.sum(inputs**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(inputs, self._embedding.weight.t()))
            
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        
        batch_size = encoding_indices.shape[0]
        encodings = torch.zeros(batch_size, self._num_embeddings, device=inputs.device)

        encodings.scatter_(1, encoding_indices, 1)
        
        quantized = torch.matmul(encodings, self._embedding.weight)
        
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        vq_loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        
        return {
            'vq_loss': vq_loss,
            'quantized': quantized, 
            'encoding_indices': encoding_indices.squeeze(1) 
        }

class VQICL(nn.Module):
    def __init__(self, args):
        super(VQICL, self).__init__()
        self.args = args
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        
        self.encoder = SASRecModel(self.item_embeddings, args) 

        self.vq_layer = VectorQuantizer(
            num_embeddings=args.num_intent_embeddings, 
            embedding_dim=args.hidden_size, 
            commitment_cost=args.commitment_cost 
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=args.hidden_size, 
            nhead=args.num_attention_heads,
            dim_feedforward=args.hidden_size * 4,
            dropout=args.hidden_dropout_prob,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=args.num_hidden_layers)
        self.prediction_head = nn.Linear(args.hidden_size, args.item_size)
        self.reconstruction_head = nn.Linear(args.hidden_size, args.item_size)
        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

    def forward(self, input_ids):
        seq_rec_logits = self.encoder(input_ids)[:, -1, :] # (B, D)
        
        vq_output = self.vq_layer(seq_rec_logits)
        vq_loss = vq_output['vq_loss']
        quantized_intent = vq_output['quantized'] # (B, D)

        intent_rep = self.prediction_head(quantized_intent) # (B, item_size)
        memory = quantized_intent.unsqueeze(1).repeat(1, self.args.max_seq_length, 1) # (B, S, D)
        
        target_emb = self.item_embeddings(input_ids)
        reconstructed_seq_hidden = self.decoder(target_emb, memory) # (B, S, D)
        recon_logits = self.reconstruction_head(reconstructed_seq_hidden) # (B, S, item_size)
        seq_aug_rep = reconstructed_seq_hidden[:, -1, :] # (B, D)

        return seq_rec_logits , recon_logits, vq_loss, seq_aug_rep


class SASRecModel(nn.Module):
    def __init__(self, item_embeddings, args):
        super(SASRecModel, self).__init__()
        self.item_embeddings = item_embeddings
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.item_encoder = Encoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args

        self.criterion = nn.BCELoss(reduction="none")
        self.apply(self.init_weights)

    def add_position_embedding(self, sequence):

        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)

        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    # model same as SASRec
    def forward(self, input_ids):

        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda == 0:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        sequence_emb = self.add_position_embedding(input_ids)

        item_encoded_layers = self.item_encoder(sequence_emb, extended_attention_mask, output_all_encoded_layers=True)

        sequence_output = item_encoded_layers[-1]
        return sequence_output

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()





