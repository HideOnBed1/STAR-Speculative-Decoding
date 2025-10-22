import argparse
import base64
from io import BytesIO
from tqdm import tqdm

parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=100)
parser.add_argument('--index', type=int, default=1)
parser.add_argument('--gpu_index', type=int, nargs='+', default=[0])
parser.add_argument('--outdir', type=str, default='outdir0')
args = parser.parse_args()
import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)[1:-1]

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from fastchat.model.model_adapter import get_conversation_template
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from torch.utils.data import Dataset, DataLoader



target_model_id = "/models/llava-v1.6-vicuna-7b-hf"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,                     
    bnb_4bit_compute_dtype=torch.float16,  
    bnb_4bit_quant_type="nf4",             
    bnb_4bit_use_double_quant=True         
)

bigmodel = LlavaNextForConditionalGeneration.from_pretrained(
    target_model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="eager",
    quantization_config=quantization_config
)
processor = AutoProcessor.from_pretrained(target_model_id)

if "vicuna" in target_model_id:
    assist_tokens = processor.tokenizer.encode("ASSISTANT:", add_special_tokens=False)
    end_tokens = processor.tokenizer.encode("ASSISTANT:", add_special_tokens=False)
    image_tokens = processor.tokenizer.encode("<image>", add_special_tokens=False)
elif "mistral" in target_model_id:
    assist_tokens = processor.tokenizer.encode("[/INST]:", add_special_tokens=False)
    end_tokens = processor.tokenizer.encode("[INST]:", add_special_tokens=False)
    image_tokens = processor.tokenizer.encode("<image>", add_special_tokens=False)

assist_len = len(assist_tokens)
end_len = len(end_tokens)

def load_image(base64_string):
    base64_string = base64_string.replace("\n", "")
    missing_padding = len(base64_string) % 4
    if missing_padding:
        base64_string += "=" * (4 - missing_padding)
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return image

def find_next_non_image_token(input_ids, image_token_ids):
    image_indices = torch.where(torch.tensor([id in image_token_ids for id in input_ids]))[0]
    
    if len(image_indices) == 0:
        return -1  
    last_image_index = image_indices[-1].item()

    for i in range(last_image_index + 1, len(input_ids)):
        if input_ids[i].item() not in image_token_ids:
            return i  

    return -1  

class MMTBenchDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        record = self.dataset[idx]
        image_base64 = record["image"]
        conversation = record["conversation"]
        image = load_image(image_base64)
        return image, conversation  
    
def collate_fn(batch):
    images, conversations = zip(*batch)
    prompts = [processor.apply_chat_template(msg, add_generation_prompt=True) for msg in conversations]
    
    return prompts, list(images)

def compute_cosine_distance(f1, f2, eps=1e-8):
    cosine_sim = F.cosine_similarity(f1, f2, dim=-1, eps=eps)  # (B, s)
    cosine_dist = 1 - cosine_sim  # (B, s)

    return cosine_dist

def compute_l2_distance(f1, f2, eps=1e-8):
    diff = f1 - f2  # (B, s, d)
    l2_dist = torch.sqrt(torch.sum(diff ** 2, dim=-1) + eps)  # (B, s)
    return l2_dist

def compute_attention_entropy(attn_weights, eps=1e-8):
    with torch.no_grad():
        attn_weights = attn_weights
        entropy = - (attn_weights * torch.log(attn_weights + eps)).sum(dim=-1)
        attn_entropy = entropy.mean(dim=1)
        del entropy  
        torch.cuda.empty_cache()
    return attn_entropy


def mid_attention_score(attn_tuple, best_layer_idx, eps=1e-8):
    L = len(attn_tuple)
    B, H, S, K = attn_tuple[0].shape

    score_list = []
    for l in range(L):
        max_per_head = attn_tuple[l].max(dim=-1).values    # (B, H, S)
        score_l = max_per_head.mean(dim=1)                 # (B, S)
        score_list.append(score_l)
    score_stack = torch.stack(score_list, dim=0).to('cpu')          # (L, B, S)

    best_layer_score = torch.gather(
        score_stack,
        dim=0,
        index=best_layer_idx.unsqueeze(0)
    ).squeeze(0)                                           # (B, S)

    top_layer_score = score_stack[-1]                     # (B, S)

    return best_layer_score, top_layer_score

def mid_feature_collect_and_score(features_tuple, attn_tuple, eps=1e-8):
    """
    Modified to use the original mid_feature_collect approach (cosine distance-based)
    instead of attention entropy approach
    """
    L = len(features_tuple)
    B, s, d = features_tuple[0].shape
    
    # Stack all hidden states
    features_stack = torch.stack(features_tuple, dim=0).to("cpu")  # (L, B, S, D)
    
    # Define candidate layers (middle 75% like original mid_feature_collect)
    candidate_start = 3
    candidate_end = int(0.75 * (L - 1))
    if candidate_end <= candidate_start:
        raise ValueError("Candidate layer range is invalid")
    
    candidate_count = candidate_end - candidate_start
    
    # Extract candidate layers and their neighbors
    candidate = features_stack[candidate_start:candidate_end, :, :, :]       # (candidate_count, B, S, D)
    candidate_prev = features_stack[candidate_start - 1:candidate_end - 1, :, :, :]  # (candidate_count, B, S, D)
    candidate_next = features_stack[candidate_start + 1:candidate_end + 1, :, :, :]  # (candidate_count, B, S, D)
    
    # Global anchor layers: first and last layer
    anchor_left = features_stack[0].unsqueeze(0)    # (1, B, S, D)
    anchor_right = features_stack[-1].unsqueeze(0)   # (1, B, S, D)
    
    def cosine_distance(x, y, eps=1e-9):
        """
        Calculate cosine distance: d(x,y) = 1 - (x · y) / (||x|| ||y|| + eps)
        """
        dot_xy = (x * y).sum(dim=-1)
        norm_x = x.norm(dim=-1)
        norm_y = y.norm(dim=-1)
        return 1 - dot_xy / (norm_x * norm_y + eps)
    
    # Calculate global distances
    global_left = cosine_distance(anchor_left, candidate)   # (candidate_count, B, S)
    global_right = cosine_distance(anchor_right, candidate)  # (candidate_count, B, S)
    global_dis = torch.abs(global_left - global_right)       # (candidate_count, B, S)
    
    # Calculate local distances
    local_left = cosine_distance(candidate_prev, candidate)   # (candidate_count, B, S)
    local_right = cosine_distance(candidate_next, candidate)  # (candidate_count, B, S)
    local_dis = torch.abs(local_left - local_right)           # (candidate_count, B, S)
    
    # Combined distance metric
    total_dis = global_dis + local_dis  # (candidate_count, B, S)
    
    # Select layer with minimum combined distance (most stable)
    best_candidate_idx = torch.argmin(total_dis, dim=0)  # (B, S)
    
    # Convert to original hidden_states layer id
    best_layer_idx = best_candidate_idx + candidate_start  # (B, S)
    
    # Extract features from the best layers
    features_stack_trans = features_stack.permute(1, 2, 0, 3)  # (B, S, L, D)
    best_layer_unsq = best_layer_idx.unsqueeze(-1)  # (B, S, 1)
    index = best_layer_unsq.unsqueeze(-1).expand(-1, -1, -1, features_stack_trans.shape[-1])
    selected_features = torch.gather(features_stack_trans, dim=2, index=index)
    selected_features = selected_features.squeeze(2)  # (B, S, D)
    
    # Calculate attention scores for the selected layers
    best_layer_scores, top_layer_scores = mid_attention_score(attn_tuple, best_layer_idx)
    
    return selected_features.cpu(), best_layer_scores.cpu(), top_layer_scores.cpu()



def filter_top_ratio_tokens(
    features: torch.Tensor,  # [B, S, D]
    scores: torch.Tensor,    # [B, S]
    ratio: float = 1
):
    import math
    B, S, D = features.shape
    K = max(1, math.ceil(S * ratio))
    topk_vals, topk_inds = torch.topk(scores, K, dim=1, largest=True, sorted=False)

    sorted_inds, _ = torch.sort(topk_inds, dim=1)  # [B, K]

    features = features.to("cpu") 
    idx_expand = sorted_inds.unsqueeze(-1).expand(-1, -1, D)  # [B, K, D]
    filtered_features = torch.gather(features, dim=1, index=idx_expand)  # [B, K, D]

    return filtered_features

def visual_token_compress(mid_image_feature, mid_image_scores, image_target_feature, target_image_scores, ratio):
    mid_compressed = filter_top_ratio_tokens(mid_image_feature, mid_image_scores, ratio)
    target_compressed = filter_top_ratio_tokens(image_target_feature, target_image_scores,ratio)
    return mid_compressed, target_compressed


def find_subsequence(tensor, subsequence):
    """Find the first occurrence of subsequence in tensor"""
    tensor_list = tensor.tolist()
    subseq_list = subsequence if isinstance(subsequence, list) else subsequence.tolist()
    
    for i in range(len(tensor_list) - len(subseq_list) + 1):
        if tensor_list[i:i+len(subseq_list)] == subseq_list:
            return i
    return -1



ds = load_dataset("ge_data/processed_data", data_files="ge_data/processed_data/train.jsonl", split="train")
batch_size = 1
ds = ds.shuffle(seed=42)
if len(ds) < args.end:
    args.end = len(ds)
ds = ds.select(range(args.start, args.end))
dataset = MMTBenchDataset(ds)
data_loader = DataLoader(
    dataset, 
    batch_size=1,
    shuffle=True, 
    num_workers=1,       
    collate_fn=collate_fn,
    pin_memory=True
)

@torch.no_grad()
def ge(data):
    print(data)
    prompts, images = data
    inputs = processor(
        images=images, 
        text=prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=5120, 
        padding_side='left'
    )
    device = bigmodel.device
    inputs = inputs.to(bigmodel.device)
    seq_length = inputs.input_ids.shape[1]
    outs_big = bigmodel(**inputs, output_hidden_states=True, output_attentions=True)
    mid_feature, _, _= mid_feature_collect_and_score(outs_big.hidden_states, outs_big.attentions)

    target_embed_full = outs_big.hidden_states[0]
    target_output_feature_full = outs_big.hidden_states[-1]


    #Get Target Score for the Instruction Tokens
    assist_position = find_subsequence(inputs.input_ids[0], assist_tokens)
    
    last_image_position = (inputs.input_ids[0] == 32000).nonzero()[-1, 0].item()
    image_start = 5
    image_end = last_image_position+1

    text_start = last_image_position+2
    text_end = assist_position + 5

    inputs_prompt= type(inputs)({
        'input_ids': inputs.input_ids[:, :text_end+1],
        'attention_mask': inputs.attention_mask[:, :text_end+1], 
        'pixel_values': inputs.pixel_values,
        'image_sizes': inputs.image_sizes
    })
    outs_big_prompt = bigmodel(**inputs_prompt, output_hidden_states=True, output_attentions=True)
    _, _, target_score_prompt = mid_feature_collect_and_score(outs_big_prompt.hidden_states, outs_big_prompt.attentions)

    
    image_attention_score = target_score_prompt[:, image_start:image_end].to(device)
    
    # Define pruning ratios from 1.0 to 0.1 with step 0.1
    pruning_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]  # [1.0, 0.9, 0.8, ..., 0.1]
    pruning_indices_list = []
    
    for ratio in pruning_ratios:
        if ratio != 1:
            # Calculate indices to keep for this ratio
            top_image_attention_rank_index = image_attention_score.topk(int((image_end - image_start) * ratio)).indices + image_start
            top_image_attention_rank_index = top_image_attention_rank_index.squeeze(0)
            keep_indexs = torch.cat((torch.arange(image_start, device=device), top_image_attention_rank_index, torch.arange(image_end, seq_length, device=device))).cpu()
            keep_indexs = keep_indexs.sort().values
            pruning_indices_list.append(keep_indexs)

        
    # Compute Loss Mask
    loss_mask = torch.zeros_like(inputs.input_ids)

    for i in range(inputs.input_ids.size(0)):
        tokens = inputs.input_ids[i]
        start_idx = None
        j = 0
        while j < tokens.size(0):
            if start_idx is None and j <= tokens.size(0) - assist_len and tokens[j:j+assist_len].tolist() == assist_tokens:
                start_idx = j 
                j += assist_len
                continue
            if start_idx is not None and j <= tokens.size(0) - end_len and tokens[j:j+end_len].tolist() == end_tokens:
                loss_mask[i, start_idx+assist_len:j] = 1
                start_idx = None 
                j += end_len  
                continue
            j += 1


    td={"loss_mask":loss_mask.cpu()}
    td["attention_mask"]=inputs.attention_mask.cpu()
    td[f"inputs_embeds"] = outs_big.hidden_states[0].cpu()
    td[f"hidden_state_mid_a"] = mid_feature.cpu()
    random_target_layer = random.choice([-1, -2, -3, -4, -5])
    td[f"target"] = outs_big.hidden_states[random_target_layer].cpu()
    td["pruning_indices"] = pruning_indices_list
    td["pruning_ratios"] = pruning_ratios

    return td

outdir = f'{args.outdir}/{args.index}'
if not os.path.exists(outdir):
    os.makedirs(outdir)

def writedata(name,data_point):
    if not os.path.exists(name):
        os.makedirs(name)
    current_length=len(os.listdir(name))
    idx=current_length
    torch.save(data_point, f'{name}/data_{idx}.ckpt')


for data in tqdm(data_loader):
    outdata = ge(data)
    writedata(outdir,outdata)


