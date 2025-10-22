<div style="display: flex; justify-content: center; align-items: center; gap: 8px; margin: 20px 0;">
  <img src="figs/logo.png" alt="STAR Logo" width="40" />
  <h1 style="margin: 0; font-size: 2em;">STAR</h1>
</div>

## Contents

- [Setup & Installation](#setup--installation)
- [Inference](#inference)
  - [With UI](#with-ui)
  - [With Code](#with-code)
- [Train](#train)
  - [Generate Train Data](#generate-train-data)
  - [Train the Auto-regression Head](#train-the-auto-regression-head)
  - [Inference on custom models](#inference-on-custom-models)
- [Evaluation](#evaluation)


## Setup & Installation


```bash
cd STAR
pip install -r requirements.txt
```

## Inference
The inference code we provide automatically allocates model weights (loading a model across multiple GPUs), allowing you to run models that exceed the memory of a single GPU.

### With UI
We have provided a suggested web interface, which you can use by running the following command. After the model is fully loaded, a URL will be output in the terminal, which you can enter into your browser to access.
```bash
python -m star.application.webui --ea-model-path /models/star-Vicuna-7B-v1.3 --base-model-path /home/apc/models/vicuna-7b-v1.3 --model-type vicuna --total-token 8
```
The *total-token* is the number of draft tokens. For smaller models and advanced GPUs, this value can be set larger. Adjusting according to the specific device and model can achieve better results. If set to -1, star will automatically configure this parameter.


## Train

### Generate Train Data
You can run the following command to generate the training data.
```bash
python -m star.ge_data.allocation --outdir [path of data]
```
### Train the Auto-regression Head

```bash
cd star/model
deepspeed main_deepspeed.py --deepspeed_config /STAR/star/train/ds_config.json --tmpdir /data/llava_vicuna_mmt_0/12_data/sharegpt_0_7999_mufp16 --cpdir /STAR/star/train/vicuna-7b-ckpt --configpath /STAR/star/train/vicuna_7B_config.json
```

## Evaluation
You can test the speed of STAR on MT-bench using the following command.
```bash
python -m star.evaluation.eval_llava\
		 --ea-model-path [path of STAR weight]\ 
		 --base-model-path [path of the original model]\
```
The above two commands will each generate a .jsonl file that records the generation results and wall time.


## Visual/Head Pruning AND TARGET FEATURE SELECTION
You can adjust visual and head pruning ratio of STAR by change the variables in `model/utils.py`
```bash
head_ratio = 1 # head pruning ratio
ratio = 0.6 # visual pruning ratio
```

You can change the last hidden state from target to last several layers by:
```bash
if idx == 31: # change 31 to one from [31,30,29,28,27] in "star/model/modeling_llava_kv.py"
```

## Acknowledgements

This project has been influenced by many excellent projects in the LLM community, such as [Medusa](https://github.com/FasterDecoding/Medusa), [EAGLE](https://github.com/SafeAILab/EAGLE), [FastChat](https://github.com/lm-sys/FastChat), and others. We first release LLaVA version, others will merge together soon.