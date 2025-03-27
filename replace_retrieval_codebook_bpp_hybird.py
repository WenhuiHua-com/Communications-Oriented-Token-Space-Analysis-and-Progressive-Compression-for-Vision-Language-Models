
import argparse
import os
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from models.blip_retrieval import blip_retrieval
import utils
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader

class CodeBook:
    def __init__(self, size, dim, device='cpu'):
        self.size = size
        self.dim = dim
        self.device = device
        self.book = torch.empty(0, dim, device=self.device) 
        self.weights = torch.empty(0, device=self.device)  
        self.counter = 0 

    def add(self, tokens, importance_scores):
        tokens = tokens.to(self.device)
        importance_scores = importance_scores.to(self.device)
        token_num, token_dim = tokens.shape

        if (self.counter + token_num) <= self.size:
            self.book = torch.cat([self.book, tokens], dim=0)
            self.weights = torch.cat([self.weights, importance_scores], dim=0)
            self.counter += token_num
        else:
            sim_matrix = torch.mm(tokens, self.book.T)
            for i, token in enumerate(tokens):
                sim_scores, indices = torch.topk(sim_matrix[i], k=1, largest=True)
                index = indices[0]

            
                wa = self.weights[index]
                wb = importance_scores[i]
                self.book[index] = (wa * self.book[index] + wb * token) / (wa + wb)
                self.weights[index] = wa + wb  

    def search(self, tokens, attention_weight_tensor, sparsity_percentage=1.0):
        tokens = tokens.to(self.device)
        # Step 1: Compute similarity between input tokens and codebook entries
        sim_matrix = torch.mm(tokens, self.book.T)  # Compute similarity between input tokens and codebook

        # Step 2: Get the most similar codebook entry for each token
        sim_scores, indices = torch.topk(sim_matrix, k=1, largest=True)  
        cls_token_copy = self.book[indices.squeeze()[0]]  # Select CLS token
        # Step 3: Sort these most similar entries based on similarity scores
        num_tokens_to_zero = int(tokens.size(0) * sparsity_percentage)  # Number of tokens to zero out
        sorted_sim_scores, sorted_indices = torch.sort(attention_weight_tensor.squeeze(), descending=False)  

        # Step 4: Select the lowest similarity entries and set them to zero
        zero_indices = sorted_indices[:num_tokens_to_zero]
        # Step 5: Replace vectors with zero where necessary
        result_vectors = self.book[indices.squeeze()]
        result_vectors[zero_indices] = 0
        result_vectors[0] = cls_token_copy  # Retain CLS token from codebook
        return result_vectors

    def save(self, path):
        torch.save({'book': self.book, 'weights': self.weights}, path)

    def load(self, path):
        data = torch.load(path)
        self.book = data['book'].to(self.device)
        self.weights = data['weights'].to(self.device)  
        self.counter = self.book.size(0)


@torch.no_grad()
def evaluation(model, data_loader, device, config, codebook):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'

    print('Computing features for evaluation...')
    start_time = time.time()

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_embeds = []
    text_atts = []
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i + text_bs)]
        text_input = model.tokenizer(text, padding='max_length', truncation=True, max_length=35,
                                     return_tensors="pt").to(device)
        text_output = model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask, mode='text')
        text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:, 0, :]))
        text_embeds.append(text_embed)
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)

    text_embeds = torch.cat(text_embeds, dim=0)
    text_ids = torch.cat(text_ids, dim=0)
    text_atts = torch.cat(text_atts, dim=0)
    text_ids[:, 0] = model.tokenizer.enc_token_id

    image_feats = []
    image_embeds = []
    class VITAttentionGradRollout:
        def __init__(self, model, attention_layer_name='attn_drop', discard_ratio=0.9):
            self.model = model
            self.discard_ratio = discard_ratio
            self.attentions = []

            # Register forward hooks on all attention layers that match the attention_layer_name
            for name, module in self.model.named_modules():
                if attention_layer_name in name:
                    module.register_forward_hook(self.get_attention)

        def get_attention(self, module, input, output):
            # Save attention output from each relevant layer
            self.attentions.append(output.cpu())

        def reset(self):
            # Clear stored attentions to avoid accumulation across multiple inputs
            self.attentions = []
    attention_extractor = VITAttentionGradRollout(model.visual_encoder)
    for image, img_id in data_loader:
        image = image.to(device)
        image_feat = model.visual_encoder(image) 

        batch_size, num_patches, embedding_dim = image_feat.shape
        image_feat_reshaped = image_feat.view(batch_size * num_patches, embedding_dim)
       
        attention_weight_list = attention_extractor.attentions
        attention_weight_tensor = torch.stack(attention_weight_list) 

        attention_weight_tensor = attention_weight_tensor.sum(dim=0).sum(dim=1).sum(dim=-1) 

        image_feat_replaced = codebook.search(image_feat_reshaped, attention_weight_tensor.squeeze(), sparsity_percentage=0.4)
        attention_extractor.reset()
        

        image_feat = image_feat_replaced.view(batch_size, num_patches, embedding_dim)  

        image_embed = model.vision_proj(image_feat[:, 0, :])
        image_embed = F.normalize(image_embed, dim=-1)

        image_feats.append(image_feat.cpu())
        image_embeds.append(image_embed)

    image_feats = torch.cat(image_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)

    sims_matrix = image_embeds @ text_embeds.t()
    score_matrix_i2t = torch.full((len(data_loader.dataset.image), len(texts)), -100.0).to(device)

    num_tasks = utils.get_world_size()
    rank = utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)

        encoder_output = image_feats[start + i].repeat(config['k_test'], 1, 1).to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        output = model.text_encoder(text_ids[topk_idx],
                                    attention_mask=text_atts[topk_idx],
                                    encoder_hidden_states=encoder_output,
                                    encoder_attention_mask=encoder_att,
                                    return_dict=True,
                                    )
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_i2t[start + i, topk_idx] = score + topk_sim

    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((len(texts), len(data_loader.dataset.image)), -100.0).to(device)

    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        encoder_output = image_feats[topk_idx].to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        output = model.text_encoder(text_ids[start + i].repeat(config['k_test'], 1),
                                    attention_mask=text_atts[start + i].repeat(config['k_test'], 1),
                                    encoder_hidden_states=encoder_output,
                                    encoder_attention_mask=encoder_att,
                                    return_dict=True,
                                    )
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_t2i[start + i, topk_idx] = score + topk_sim

    if args.distributed:
        dist.barrier()
        torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()


@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    # Images->Text
    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])

    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result = {'txt_r1': tr1,
                   'txt_r5': tr5,
                   'txt_r10': tr10,
                   'txt_r_mean': tr_mean,
                   'img_r1': ir1,
                   'img_r5': ir5,
                   'img_r10': ir10,
                   'img_r_mean': ir_mean,
                   'r_mean': r_mean}
    return eval_result


def main(args, config):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    codebook_size = 524288
    embedding_dim = 768  
    codebook = CodeBook(size=codebook_size, dim=embedding_dim, device=device)
    codebook.load('/media/data/huawenhui/BLIP-main/BLIP-main/codebook_retrieval_New_10000_524288_384_0.5.pth')
    print('loading codebook sets')
    #### Dataset ####
    print("Creating retrieval dataset")
    print('Cos search')
    train_dataset, val_dataset, test_dataset = create_dataset('retrieval_%s' % config['dataset'], config)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
    else:
        samplers = [None, None, None]

    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset], samplers,
                                                          batch_size=[config['batch_size_train']] +
                                                                     [1] * 2,
                                                          num_workers=[4, 4, 4],
                                                          is_trains=[True, False, False],
                                                          collate_fns=[None, None, None])
    print(f'\n Num_train : {len(train_loader)} Num_val : {len(val_loader)} Num_test : {len(test_loader)}')

    #### Model ####
    print("Creating model")
    model = blip_retrieval(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'],
                           vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'],
                           queue_size=config['queue_size'], negative_all_rank=config['negative_all_rank'])

    model = model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    print("Start eval")
    start_time = time.time()
    
    score_test_i2t, score_test_t2i = evaluation(model_without_ddp, test_loader, device, config, codebook)

    test_result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img,
                           test_loader.dataset.img2txt)
    print(f'\n test_result : {test_result}')

    log_stats = {
        **{f'test_{k}': v for k, v in test_result.items()},
    }
    with open(os.path.join(args.output_dir, "evaluate.txt"), "a") as f:
        f.write(json.dumps(log_stats) + "\n")

        # dist.barrier()
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Eval time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/retrieval_flickr.yaml')
    parser.add_argument('--output_dir', default='output/Retrieval_flickr')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)
