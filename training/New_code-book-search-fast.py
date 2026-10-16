import os
import torch
from torch.nn import functional as F
from PIL import Image
import numpy as np
import random
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from models.blip_itm import blip_itm
from models.blip import blip_decoder
from models.blip_retrieval import blip_retrieval

def compute_kl_divergence(p, q):
    p = p + 1e-10  # Avoid log(0).
    q = q + 1e-10
    return (p * (p.log() - q.log())).sum()

import torch
import torch.nn.functional as F

def importance_function(image_embeds, model, input_ids, model_kwargs, sparsity_percentage):
    num_tokens = image_embeds.size(1)
    num_tokens_to_keep = int(num_tokens * (1 - sparsity_percentage))
    mask = torch.ones(num_tokens, dtype=torch.bool).to(image_embeds.device)
  # Compute the reference decoder distribution.
    with torch.no_grad():
        original_decoder_output = model.text_decoder(input_ids=input_ids, **model_kwargs)
        original_logits = original_decoder_output.logits
        original_probabilities = F.softmax(original_logits, dim=-1)

    kl_divergences = torch.zeros(num_tokens, dtype=torch.float).to(image_embeds.device)

    for i in range(num_tokens):
        temp_mask = mask.clone()
        temp_mask[i] = False

        temp_image_embeds = image_embeds[:, temp_mask, :]
        temp_image_atts = torch.ones(temp_image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)
        temp_model_kwargs = {"encoder_hidden_states": temp_image_embeds, "encoder_attention_mask": temp_image_atts}

        with torch.no_grad():
            temp_decoder_output = model.text_decoder(input_ids=input_ids, **temp_model_kwargs)
            temp_logits = temp_decoder_output.logits
            temp_probabilities = F.softmax(temp_logits, dim=-1)

        kl_divergences[i] = compute_kl_divergence(original_probabilities, temp_probabilities)

  # Remove tokens with the lowest importance scores.
    _, indices_to_remove = torch.topk(kl_divergences, num_tokens - num_tokens_to_keep, largest=False)

    mask[indices_to_remove] = False

  # Retain the selected tokens and their importance scores.
    sparse_image_embeds = image_embeds[:, mask, :]
    sparse_kl_divergencess = kl_divergences[mask]
    importance_scores = F.softmax(sparse_kl_divergencess, dim=0)
    return sparse_image_embeds, importance_scores

def print_progress(count, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    percent = (count / total) * 100
    filled_length = int(length * count // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    progress_str = f'{percent:.{decimals}f}%'
    print(f'\r{prefix} |{bar}| {progress_str} {suffix}', end='', flush=True)

def load_demo_image(image_size, device, picture_name, image_root):
    raw_image = Image.open(os.path.join(image_root, picture_name)).convert('RGB')

    w, h = raw_image.size

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    image = transform(raw_image).unsqueeze(0).to(device)
    return image

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

val_image_root = '/media/data/huawenhui/mae-main/mae-main/data/val'
image_name_list = []
for i in os.listdir(val_image_root):
    for j in os.listdir(os.path.join(val_image_root, i)):
        image_name_list.append(os.path.join(os.path.join(val_image_root, i), j))

train_image_root = '/media/data/huawenhui/mae-main/mae-main/data/train'
train_image_name_list = []
for i in os.listdir(train_image_root):
    for j in os.listdir(os.path.join(train_image_root, i)):
        train_image_name_list.append(os.path.join(os.path.join(train_image_root, i), j))

if len(train_image_name_list) > 5000:
    train_image_name_list = random.sample(train_image_name_list, 20000)

image_size = 224
model_url = '/media/data/huawenhui/BLIP-main/BLIP-main/model_base_capfilt_large.pth'
score_url = '/media/data/huawenhui/BLIP-main/BLIP-main/model_base_retrieval_coco.pth'
model = blip_decoder(pretrained=model_url, image_size=image_size, vit='base')
score_model = blip_itm(pretrained=score_url, image_size=image_size, vit='base')
model.eval()
model = model.to(device)
score_model.eval()
score_model.to(device)

model_retrieval = blip_retrieval(pretrained='/media/data/huawenhui/BLIP-main/BLIP-main/model_base_retrieval_flickr.pth',
                                 image_size=384, vit='base',
                                 vit_grad_ckpt=True, vit_ckpt_layer=4,
                                 queue_size=57600, negative_all_rank=False)

model_retrieval = model_retrieval.to(device)

results = {}

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

  # Importance-weighted aggregation.
                wa = self.weights[index]
                wb = importance_scores[i]
                self.book[index] = (wa * self.book[index] + wb * token) / (wa + wb)
                self.weights[index] = wa + wb  # Accumulate the merged weight.

    def search(self, tokens):
        tokens = tokens.to(self.device)
        sim_matrix = torch.mm(tokens, self.book.T)
        _, indices = torch.topk(sim_matrix, k=1, largest=True)
        return indices.squeeze()

    def save(self, path):
        torch.save({'book': self.book, 'weights': self.weights}, path)

    def load(self, path):
        data = torch.load(path)
        self.book = data['book'].to(self.device)
        self.weights = data['weights'].to(self.device)
        self.counter = self.book.size(0)

def add_to_codebook(image_embeds, importance_scores, codebook):

    codebook.add(image_embeds.squeeze(0), importance_scores.squeeze())

def test_codebook(image, model, codebook):
    sparse_image_embeds = model.visual_encoder(image)
    token_indices = codebook.search(sparse_image_embeds.squeeze(0))
    retrieved_tokens = codebook.book[token_indices]
    return retrieved_tokens

def save_tokens_and_scores(image_token, importance_score, token_path, score_path):
    torch.save(image_token, token_path)
    torch.save(importance_score, score_path)

def load_tokens_and_scores(token_path, score_path):
    image_token = torch.load(token_path)
    importance_score = torch.load(score_path)
    return image_token, importance_score

codebook_size = 524288
embedding_dim = 768
codebook = CodeBook(size=codebook_size, dim=embedding_dim, device=device)
sparsity_percentage = 0.5

# Extract tokens and importance scores.
count = 0
cos = []
image_token = []
importance_score = []
"""
Extract sample features and importance scores
"""
for picture_name in train_image_name_list:
    with torch.no_grad():

        ori_image = load_demo_image(image_size=image_size, device=device, picture_name=picture_name,
                                    image_root=train_image_root)

        image_embeds = model.visual_encoder(ori_image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(ori_image.device)
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask": image_atts}
        prompt = ['a picture of '] * ori_image.size(0)
        input_ids = model.tokenizer(prompt, return_tensors="pt").input_ids.to(ori_image.device)
        input_ids[:, 0] = model.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1]
        count += 1
        image_embeds, importance_ = importance_function(image_embeds, model, input_ids, model_kwargs, sparsity_percentage)

        image_token.append(image_embeds.squeeze(0))
        importance_score.append(importance_)

        if count % 2500 == 0:
            print_progress(count, len(train_image_name_list), prefix='Progress:', suffix='Complete', length=30)

    del ori_image, image_embeds, image_atts, input_ids, importance_
    torch.cuda.empty_cache()

save_tokens_and_scores(image_token, importance_score, 'image_token_0.5.pth', 'importance_score_0.5.pth')

# Load cached tokens and importance scores, then train the codebook.
image_token, importance_score = load_tokens_and_scores('image_token_0.5.pth', 'importance_score_0.5.pth')

for tokens, scores in zip(image_token, importance_score):
    add_to_codebook(tokens.unsqueeze(0), scores, codebook)
# Save the codebook.
codebook.save('codebook_retrieval_New_复现.pth')
# Evaluate the codebook.
codebook.load('codebook_retrieval_New_复现.pth')
print('Training Finished')
# Evaluate the codebook.
count = 0
for picture_name in image_name_list:
    with torch.no_grad():
        ori_image = load_demo_image(image_size=image_size, device=device, picture_name=picture_name,
                                    image_root=val_image_root)
        retrieved_tokens = test_codebook(ori_image, model_retrieval, codebook)

        caption = model.feature_generate(retrieved_tokens.unsqueeze(0))
        itc_score = score_model(ori_image, caption, match_head='itc')
        cos.append(itc_score.detach().cpu().numpy())
        count += 1
        if count % 1000 == 0:
            print(picture_name, count, caption)
    del ori_image, retrieved_tokens
    torch.cuda.empty_cache()

print(f'\nSparsity {sparsity_percentage * 100}%: Cosine Similarity = {np.array(cos).mean()}')
