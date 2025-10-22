import argparse
import deepspeed

parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--basepath', type=str, default='models/llava-v1.6-vicuna-7b-hf')
parser.add_argument('--configpath', type=str, default="train/vicuna_7B_config.json")
parser.add_argument('--tmpdir', type=str,
                    default='/ablation/fix')

parser.add_argument('--cpdir', type=str, default='/model/ablation/llava_l')
parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()
import json
import wandb

wandb.login()


train_config = {
    "lr": 5e-5,
    "bs": 4,
    "gradient_accumulation_steps": 1,
    "datapath": f"{args.tmpdir}",
    "is_warmup": True,
    "num_epochs": 20,
    "num_warmup_steps": 2000,
    "total_steps": 800000,
    "p_w": 1.0,
    "v_w": 0.2,
    "head_w": 0.1,
    "num_workers": 1,
    "embeding": True,
    "act": "No",
    "data_noise": True,
    "noise": "uniform",
    "mean": 0.0,
    "std": 0.2,
    "residual": "true,norm",
    "max_len": 5120,
    "config_path": args.configpath,
    "b1": 0.9,
    "b2": 0.95,
    "grad_clip": 0.5,
}

from safetensors import safe_open
import os
import torch

torch.backends.cuda.matmul.allow_tf32 = True
from accelerate import Accelerator
from accelerate.utils import set_seed

set_seed(0)
accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=train_config["gradient_accumulation_steps"])


from cnets import Model
from configs import EConfig
from typing import Any, Dict, List

from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
# import accelerate
from torch.autograd import Variable
import numpy as np


deepspeed.init_distributed()
rank = torch.distributed.get_rank()
if rank == 0:
    import wandb
    wandb.init(project="training_ofa", name=f"attention_training_{train_config['lr']}")
try:
    with open(os.path.join(args.basepath, "model.safetensors.index.json"), "r") as f:
        index_json = json.loads(f.read())
        head_path = index_json["weight_map"]["language_model.lm_head.weight"]
    with safe_open(os.path.join(args.basepath, head_path),
                   framework="pt",
                   device="cpu") as f:
        tensor_slice = f.get_slice("language_model.lm_head.weight")
        vocab_size, hidden_dim = tensor_slice.get_shape()
        tensor = tensor_slice[:, :hidden_dim].float()
except:
    with open(os.path.join(args.basepath, "pytorch_model.bin.index.json"), "r") as f:
        index_json = json.loads(f.read())
        head_path = index_json["weight_map"]["lm_head.weight"]
    weights = torch.load(os.path.join(args.basepath, head_path))
    tensor = weights["lm_head.weight"].float()
    

head = torch.nn.Linear(tensor.shape[1], tensor.shape[0], bias=False)
head.weight.data = tensor


def list_files(path):
    datapath = []
    for root, directories, files in os.walk(path, followlinks=True):
        for file in files:
            file_path = os.path.join(root, file)
            datapath.append(file_path)
    return datapath


class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.0):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise = torch.randn(tensor.size()) * self.std + self.mean
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data


class AddUniformNoise:
    def __init__(self, std=0.0):
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise = (torch.rand_like(tensor) - 0.5) * self.std * 512 / tensor.shape[1]
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data


class CustomDataset(Dataset):
    def __init__(self, datapath, transform=None):
        self.data = datapath
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = torch.load(self.data[index])
        new_data = {}

        hidden_state = data['target'][:train_config["max_len"]]
        hidden_state_second = hidden_state.clone()
        inputs_embeds = data['inputs_embeds'][:train_config["max_len"]]
        hidden_state_mid = data['hidden_state_mid_a'][:train_config["max_len"]]
        loss_mask = data["loss_mask"][:train_config["max_len"]]
        indices =  data["pruning_indices"]
        ratios = data["pruning_ratios"]


        length = hidden_state.shape[1]
        attention_mask = [1] * length
        loss_mask = loss_mask[0].tolist()

        target = hidden_state
        
        hidden_state = hidden_state[:,:-1,:]
        new_data["attention_mask"] = attention_mask
        new_data["loss_mask"] = loss_mask
        new_data["target"] = target
        new_data["hidden_state_big"] = hidden_state
        new_data["hidden_state_mid"] = hidden_state_mid
        new_data["inputs_embeds"] = inputs_embeds
        new_data["ratios"] = ratios
        new_data["indices"] = indices 


        if self.transform:
            new_data = self.transform(new_data)

        return new_data


class DataCollatorWithPadding:

    def paddingtensor(self, intensors, N):
        B, n, S = intensors.shape
        padding_tensor = torch.zeros(B, N - n, S)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item['hidden_state_mid'].shape[1] for item in features)
        
        batch_inputs_embeds = torch.cat([self.paddingtensor(item['inputs_embeds'], max_length) for item in features])
        batch_hidden_states = torch.cat([self.paddingtensor(item['hidden_state_big'], max_length-1) for item in features])
        batch_hidden_states_mid = torch.cat([self.paddingtensor(item['hidden_state_mid'], max_length) for item in features])
        batch_target = torch.cat([self.paddingtensor(item['target'], max_length) for item in features])
        batch_loss_mask = torch.tensor(
            [item['loss_mask'] + [0] * (max_length - len(item['loss_mask'])) for item in features])
        batch_attention_mask = torch.tensor(
            [item['attention_mask'] + [0] * (max_length - len(item['attention_mask'])) for item in features])
        batch_indices = []
        for item in features:
            batch_indices.append(item["indices"])

        batch = {
            #"input_ids": batch_input_ids,
            "inputs_embeds": batch_inputs_embeds,
            "hidden_states": batch_hidden_states,
            "hidden_states_mid": batch_hidden_states_mid,
            "target": batch_target,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
            "indices": batch_indices ,
            "ratios": features[0]["ratios"],
            "max_length": max_length 
        }
        return batch

def top_accuracy(output, target, topk=(1,)):
    # output.shape (bs, num_classes), target.shape (bs, )
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res

def compute_loss(target, target_p, predict, loss_mask):
    out_head = head_engine(predict)
    out_logp = nn.LogSoftmax(dim=2)(out_head)
    plogp = target_p * out_logp
    ploss = -torch.sum(torch.sum(loss_mask * plogp, 2)) / (loss_mask.shape[0] * loss_mask.shape[1] + 1e-5)
    vloss = criterion(predict, target.to(rank))
    vloss = torch.sum(torch.mean(loss_mask * vloss, 2)) / (loss_mask.shape[0] * loss_mask.shape[1] + 1e-5)
    kldloss = F.kl_div(out_logp, target_p, reduction='none')
    kldloss = torch.sum(torch.mean(loss_mask * kldloss, 2)) / (loss_mask.sum() + 1e-5)
    return vloss, kldloss, out_head

def compute_mid_loss(target, predict, loss_mask):
    vloss = criterion(predict, target.to(rank))
    vloss = torch.sum(torch.mean(loss_mask * vloss, 2)) / (loss_mask.shape[0] * loss_mask.shape[1] + 1e-5)
    return vloss

if train_config["data_noise"]:
    if train_config["noise"] == "uniform":
        aug = AddUniformNoise(std=train_config["std"])
    else:
        aug = AddGaussianNoise(mean=train_config["mean"], std=train_config["std"])
else:
    aug = None

datapath = list_files(train_config["datapath"])
traindatapath = datapath[:int(len(datapath) * 1)]
testdatapath = datapath[int(len(datapath) * 1):]
traindataset = CustomDataset(traindatapath, transform=aug)
testdataset = CustomDataset(testdatapath)
test_loader = DataLoader(testdataset, batch_size=train_config["bs"], shuffle=False,
                         collate_fn=DataCollatorWithPadding(), num_workers=train_config["num_workers"], pin_memory=True)


if rank == 0:
    if not os.path.exists(args.cpdir):
        os.makedirs(args.cpdir)

config = EConfig.from_pretrained(train_config["config_path"])
model = Model(config, path=args.basepath, load_emb=True)

criterion = nn.SmoothL1Loss(reduction="none")

num_epochs = train_config["num_epochs"]
num_warmup_steps = train_config["num_warmup_steps"]
total_steps = train_config["total_steps"]
is_warmup = train_config["is_warmup"]

model_engine, optimizer, train_loader, _ = deepspeed.initialize(args=args,
                                                                model=model,
                                                                model_parameters=model.parameters(),
                                                                training_data=traindataset,
                                                                collate_fn=DataCollatorWithPadding()
                                                                )

head_engine, _, test_loader, _ = deepspeed.initialize(args=args,
                                                      model=head,
                                                      model_parameters=head.parameters(),
                                                      training_data=testdataset,
                                                      collate_fn=DataCollatorWithPadding()
                                                      )

import random
random.seed(42) 


def pruning_for_data(data):
    pruning_item_idx = random.randint(0, len(data["ratios"]) - 1)
    pruning_ratio = data["ratios"][pruning_item_idx]
    batch_size = data["inputs_embeds"].shape[0]

    if pruning_ratio == 1:
        return data

    new_data = {}
    pruned_inputs_embeds = []
    pruned_hidden_states = []
    pruned_hidden_states_mid = []
    pruned_target = []
    pruned_attention_mask = []
    pruned_loss_mask = []

    keep_indices = data["indices"][0][pruning_item_idx]
    DEVICE = data["inputs_embeds"].device

    pruned_inputs_embeds.append(data["inputs_embeds"][:, keep_indices, :])
    pruned_hidden_states.append(data["hidden_states"][:, keep_indices[:-1], :])  # Exclude last for hidden_states
    pruned_hidden_states_mid.append(data["hidden_states_mid"][:, keep_indices, :])
    pruned_target.append(data["target"][:, keep_indices, :])
    pruned_attention_mask.append(data["attention_mask"][:, keep_indices])
    pruned_loss_mask.append(data["loss_mask"][:, keep_indices])

    new_data["inputs_embeds"] = torch.cat(pruned_inputs_embeds, dim=0).to(DEVICE)
    new_data["hidden_states"] = torch.cat(pruned_hidden_states, dim=0).to(DEVICE)
    new_data["hidden_states_mid"] = torch.cat(pruned_hidden_states_mid, dim=0).to(DEVICE)
    new_data["target"] = torch.cat(pruned_target, dim=0).to(DEVICE)
    new_data["attention_mask"] = torch.cat(pruned_attention_mask, dim=0).to(DEVICE)
    new_data["loss_mask"] = torch.cat(pruned_loss_mask, dim=0).to(DEVICE)


    return new_data


for param in head.parameters():
    param.requires_grad = False


# checkpoint_dir = "/ofa_2main_2_cf_head_mix_final/state_6"
# _, client_state = model_engine.load_checkpoint(checkpoint_dir)

for epoch in range(0, num_epochs):
    top_3acc = [0 for _ in range(3)]
    correct = 0
    total = 0
    epoch_loss = 0
    num_batches = 0
    model.train()
    for batch_idx, data in enumerate(tqdm(train_loader)):

        model.zero_grad()

        last_head_ratio = 1

    
        if epoch > 1:
            data = pruning_for_data(data)
            last_head_ratio = random.choice([0.25, 0.5, 0.75, 1])
        else: 
            last_head_ratio = 1

        inputs_embeds = Variable(data["inputs_embeds"].to(torch.bfloat16), requires_grad=True)
        last_hidden_state = Variable(data["hidden_states"].to(torch.bfloat16), requires_grad=True)


        predict, all_hidden_states = model_engine(last_hidden_state.to(rank), inputs_embeds=inputs_embeds.to(rank),
                               attention_mask=data["attention_mask"].to(rank), output_hidden_states=True, last_head_ratio=last_head_ratio)

        mid_predict = all_hidden_states[-2]
        with torch.no_grad():
            target_head = head_engine(data["target"].to(rank).to(torch.bfloat16))
            target_p = nn.Softmax(dim=2)(target_head)
            target_p = target_p.detach()

        loss_mask = data["loss_mask"][:, :, None].to(rank)
        vloss, ploss, out_head = compute_loss(data["target"], target_p, predict, loss_mask)
        mid_vloss = compute_mid_loss(data["hidden_states_mid"], mid_predict, loss_mask)
        loss = train_config["v_w"] * vloss + train_config["p_w"] * ploss + mid_vloss
        model_engine.backward(loss)
        model_engine.step()

        with torch.no_grad():
            _, predicted = torch.max(out_head, 2)
            _, target = torch.max(target_head, 2)
            ct = loss_mask.sum().item()
            cc = ((predicted == target) * loss_mask.squeeze()).sum().item()
            out_head = out_head.view(-1, target_head.shape[-1])[loss_mask.view(-1) == 1]
            target = target.view(-1)[loss_mask.view(-1) == 1]
            topkacc = top_accuracy(out_head, target, (1, 2, 3))
            for top_i in range(len(topkacc)):
                top_3acc[top_i] += topkacc[top_i]
            total += ct
            correct += cc
        if rank == 0 and ct != 0:
            logdict = {"train/lr": optimizer.optimizer.param_groups[0]["lr"], "train/vloss": vloss.item(),
                       "train/ploss": ploss.item(), "train/loss": loss.item(), "train/acc": cc / ct}
            for id, i in enumerate(top_3acc):
                logdict[f'train/top_{id + 1}_acc'] = topkacc[id].item() / ct
            for id,i in enumerate(top_3acc):
                 wandb.log({f'train/top_{id+1}_acc':topkacc[id].item()/ct})

        del ploss, vloss
        epoch_loss += loss.item()
        num_batches += 1

    correct, total = torch.tensor(correct).cuda(), torch.tensor(total).cuda()
    correct, total = accelerator.gather_for_metrics((correct, total))
    correct, total = correct.sum().item(), total.sum().item()
    epoch_loss /= num_batches
    top_3acc = accelerator.gather_for_metrics(top_3acc)
    if accelerator.is_local_main_process:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
        print('Train Accuracy: {:.2f}%'.format(100 * correct / (total + 1e-5)))
        wandb.log(
            {"train/epochacc": correct / (total + 1e-5), "train/epochloss": epoch_loss})

    model_engine.save_16bit_model(f"{args.cpdir}/state_{epoch}")
    deepspeed.DeepSpeedEngine.save_checkpoint(model_engine, save_dir=f"{args.cpdir}/state_{epoch}")
