#  Effective-llm-jailbreak


Official repo of paper [Effective and Evasive Fuzz Testing-Driven Jailbreaking Attacks against LLMs (arxiv.org)](https://arxiv.org/abs/2409.14866)

## Overview

![overview.png](./overview.png)

## Installation

```bash
python = 3.10
pytorch = 2.1.2+cu12.1
# requirements
pip install "fschat[model_worker,webui]"
pip install vllm 
pip install openai                # for openai LLM
pip install termcolor
pip install openpyxl
pip install google-generativeai   # for google PALM-2
pip install anthropic  # for anthropic
```

## Models

1. We use a finetuned RoBERTa-large model [huggingface](https://huggingface.co/hubert233/GPTFuzz) from [GPTFuzz](https://github.com/sherdencooper/GPTFuzz) as our judge model. Thanks to its great work!

2. For Judge model, we need to set api-key for gpt judge model:

   ```python
   # line 106 in ./Judge/language_models.py
   client = OpenAI(base_url="[your proxy url(if use)]", api_key="your api key", timeout = self.API_TIMEOUT)
   ```

   

## Datasets

We have 3 avaible datasets to jailbreak:

1. `datasets/questions/question_target_list.csv` : sampled from two public datasets: [llm-jailbreak-study](https://sites.google.com/view/llm-jailbreak-study) and [hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf). Following the format of [GCG](https://github.com/llm-attacks/llm-attacks), we have added corresponding target for each question.
2. `datasets/questions/question_target.csv`  : advbench.

3. `datasets/questions/question_target_custom.csv`  : subset of advbench.

## Example to use

to jailbreak gpt-3.5-turbo on the subset of advbench:

```bash
python run.py --openai_key [your openai_key] --model_path gpt-3.5-turbo --target_model gpt-3.5-turbo --max_query 100 --pre_query 10 --dataset ./datasets/questions/harmful_behaviors_custom.csv 
```
## eval
set '''directory_path''' as the directory of result, then '''run eval.py''' to get the ASR and AQ.

## Acknowledgement

This code is developed heavily relying on [GPTFuzz](https://github.com/Fictionarry/ER-NeRF), and also [PAIR](https://github.com/patrickrchao/jailbreakingllms). Thanks to these excellent works!

## Citation

Please kindly cite our paper:

```
@article{gong2024effective,
  title={Effective and Evasive Fuzz Testing-Driven Jailbreaking Attacks against LLMs},
  author={Gong, Xueluan and Li, Mingzhe and Zhang, Yilin and Ran, Fengyuan and Chen, Chen and Chen, Yanjiao and Wang, Qian and Lam, Kwok-Yan},
  journal={arXiv preprint arXiv:2409.14866},
  year={2024}
}
```

