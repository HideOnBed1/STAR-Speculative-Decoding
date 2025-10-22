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

target_model_id = "llava-v1.6-vicuna-7b-hf"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,                    
    bnb_4bit_compute_dtype=torch.float16, 
    bnb_4bit_quant_type="nf4",             
    bnb_4bit_use_double_quant=True        
)

bigmodel = LlavaNextForConditionalGeneration.from_pretrained(
    target_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
   # quantization_config=quantization_config
)
processor = AutoProcessor.from_pretrained(target_model_id)

assist_tokens = processor.tokenizer.encode("ASSISTANT:", add_special_tokens=False)
end_tokens = processor.tokenizer.encode("ASSISTANT:", add_special_tokens=False)
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
    print(conversations)

    prompts = [processor.apply_chat_template(msg, add_generation_prompt=True) for msg in conversations]
    
    return prompts, list(images)

ds = load_dataset("", data_files="mmt-bench-llava-v1.6-vicuna-7b.jsonl", split="train")
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
    num_workers=4,        
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
    inputs = inputs.to("cuda:0")
    print(processor.tokenizer.decode(inputs.input_ids[0]))
    outs_big = bigmodel(**inputs, output_hidden_states=True)
    assert len(outs_big.hidden_states) == 33
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
                print(loss_mask[i], start_idx, j)
                start_idx = None  
                j += end_len  
                continue
            j += 1
            

    td={"loss_mask":loss_mask.cpu()}
    td["attention_mask"]=inputs.attention_mask.cpu()
    # early exit layer 
    # exit at layer2 for vicuna-7B and layer3 for vicuna-13B 
    td[f"inputs_embeds"] = outs_big.hidden_states[0].cpu()
    td[f"hidden_state_layer2"] = outs_big.hidden_states[2].cpu()
    td[f"hidden_state_layer4"] = outs_big.hidden_states[4].cpu()
    td[f"hidden_state_layer8"] = outs_big.hidden_states[8].cpu()
    td[f"hidden_state_layer12"] = outs_big.hidden_states[12].cpu()
    td[f"hidden_state_layer24"] = outs_big.hidden_states[24].cpu()
    td[f"target"] = outs_big.hidden_states[-1].cpu()
    pad_index = find_next_non_image_token(inputs.input_ids[0], image_tokens)
    zeros_column = torch.zeros(outs_big.hidden_states[-1].shape[0], 1, outs_big.hidden_states[-1].shape[2], device=outs_big.hidden_states[-1].device)
    td[f"hidden_state"] = torch.cat(
    (
        zeros_column,
        outs_big.hidden_states[-1][:, :-1, :]
    ),
    dim=1).cpu()
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


