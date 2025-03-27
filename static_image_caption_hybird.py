import os
import torch
from torch.nn import functional as F
from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from models.blip_itm import blip_itm
from models.blip import blip_decoder
import json
from models.blip_retrieval import blip_retrieval
with open('/media/data/huawenhui/BLIP-main/BLIP-main/flick/annotation/flickr30k_test.json', 'r') as file:
    data = json.load(file)  # Read json for test
class CodeBook:
    def __init__(self, size, dim, device='cpu'):
        self.size = size
        self.dim = dim
        self.device = device
        self.book = torch.empty(0, dim, device=self.device)
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

                # 权重加权聚合
                wa = self.weights[index]
                wb = importance_scores[i]
                self.book[index] = (wa * self.book[index] + wb * token) / (wa + wb)
                self.weights[index] = wa + wb  

    def search(self, tokens, attention_weight_tensor, sparsity_percentage=1.0):
        tokens = tokens.to(self.device)
        # Step 1: Compute similarity between input tokens and codebook entries
        sim_matrix = torch.mm(tokens, self.book.T)  # Compute similarity between input tokens and codebook

        # Step 2: Get the most similar codebook entry for each token
        sim_scores, indices = torch.topk(sim_matrix, k=1, largest=True)  # 获取相似度分数和索引
        cls_token_copy = self.book[indices.squeeze()[0]]  # Select CLS token
        # Step 3: Sort these most similar entries based on similarity scores
        num_tokens_to_zero = int(tokens.size(0) * sparsity_percentage)  # Number of tokens to zero out
        sorted_sim_scores, sorted_indices = torch.sort(attention_weight_tensor.squeeze(), descending=False)  # 按相似度升序排序

        # Step 4: Select the lowest similarity entries and set them to zero
        zero_indices = sorted_indices[:num_tokens_to_zero]
        # Step 5: Replace vectors with zero where necessary
        result_vectors = self.book[indices.squeeze()]
        result_vectors[zero_indices] = 0
        result_vectors[0] = cls_token_copy  # Retain CLS token from codebook
        result_vectors_new = [token for token in result_vectors if token.sum() != 0]
        result_vectors_new = torch.stack(result_vectors_new)  
        return result_vectors_new

    def save(self, path):
        torch.save({'book': self.book, 'weights': self.weights}, path)

    def load(self, path):
        data = torch.load(path)
        self.book = data['book'].to(self.device)
        self.weights = data['weights'].to(self.device)  
        self.counter = self.book.size(0)


def test_codebook(ori_image, model, codebook, attention_extractor, sparsity_percentages):
    sparse_image_embeds = model.visual_encoder(ori_image)
    attention_weight_list = attention_extractor.attentions  
    attention_weight_tensor = torch.stack(attention_weight_list)  
    attention_weight_tensor = attention_weight_tensor.sum(dim=0).sum(dim=1).sum(dim=-1) 
    token_indices = codebook.search(sparse_image_embeds.squeeze(0), attention_weight_tensor.squeeze(), sparsity_percentages)
    attention_extractor.reset()
    return token_indices


def load_demo_image(image_size, device, picture_name, image_root):
    raw_image = Image.open(os.path.join(image_root, picture_name)).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    image = transform(raw_image).unsqueeze(0).to(device)
    return image


image_name_list = []
for item in data:
    image_name_list.append(item['image'])  # Add the image filename to the list
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
val_image_root = '/media/data/huawenhui/BLIP-main/BLIP-main/flick/flickr30k-images'

image_size = 384
model_url = '/media/data/huawenhui/BLIP-main/BLIP-main/model_base_capfilt_large.pth'
score_url = '/media/data/huawenhui/BLIP-main/BLIP-main/model_base_retrieval_coco.pth'
model = blip_decoder(pretrained=model_url, image_size=image_size, vit='base')
score_model = blip_itm(pretrained=score_url, image_size=image_size, vit='base')
model.eval()
model = model.to(device)
score_model.eval()
score_model.to(device)

codebook_size = 524288
embedding_dim = 768
codebook = CodeBook(size=codebook_size, dim=embedding_dim, device=device)
codebook.load('/media/data/huawenhui/BLIP-main/BLIP-main/codebook_retrieval_New_10000_524288_384_0.5.pth')
print('Loading codebook sets')

model_retrieval = blip_retrieval(pretrained='/media/data/huawenhui/BLIP-main/BLIP-main/model_base_retrieval_flickr.pth',
                                 image_size=384, vit='base',
                                 vit_grad_ckpt=True, 
                                 queue_size=57600, negative_all_rank=False)

model_retrieval = model_retrieval.to(device)
model_retrieval.eval()
sparsity_percentage = [0.2, 0.4]
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
attention_extractor = VITAttentionGradRollout(model_retrieval.visual_encoder)

for sparsity_percentages in sparsity_percentage:
    captions = []  # To store all captions for this sparsity level
    count = 0
    cos = []
    for picture_name in image_name_list:
        with torch.no_grad():
            ori_image = load_demo_image(image_size=image_size, device=device, picture_name=picture_name,
                                        image_root=val_image_root)
            retrieved_tokens = test_codebook(ori_image, model_retrieval, codebook, attention_extractor, sparsity_percentages)
            caption = model.feature_generate(retrieved_tokens.unsqueeze(0))
            captions.append(caption[0])  # Save caption with image name


            count += 1

            if count % 100 == 0:
                print(picture_name, count, caption)

        del ori_image, retrieved_tokens
        torch.cuda.empty_cache()

    # Calculate BPP based on sparsity percentage
    BPP = ((1 - sparsity_percentages) * 576 * 19) / (384 * 384)

    # Save the captions to a text file with BPP in the filename
    filename = f"BPP_{BPP:.4f}_Hybird.txt"
    with open(filename, "w") as f:
        for caption in captions:
            f.write(f"{caption}\n")

    
