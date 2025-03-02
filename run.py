import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # for debugging
import csv
from fastchat.model import add_model_args
import argparse
import pandas as pd
from gptfuzzer.fuzzer.selection import MCTSExploreSelectPolicy
from gptfuzzer.fuzzer.mutator import (
    MutateRandomSinglePolicy, OpenAIMutatorExpand, OpenAIMutatorscenario, OpenAIMutatorcharacters)
from gptfuzzer.fuzzer import GPTFuzzer
from gptfuzzer.llm import OpenAILLM, LocalVLLM, LocalLLM, PaLM2LLM, ClaudeLLM
from gptfuzzer.utils.predict import RoBERTaPredictor
import random
random.seed(100)
import logging
httpx_logger: logging.Logger = logging.getLogger("httpx")
# disable httpx logging
httpx_logger.setLevel(logging.WARNING)

def main(args):
    initial_seed = pd.read_csv(args.seed_path)['text'].tolist()

    openai_model = OpenAILLM(args.model_path, args.openai_key)
    # target_model = PaLM2LLM(args.target_model, args.palm_key)
    # target_model = ClaudeLLM(args.target_model, args.claude_key)
    # target_model = OpenAILLM(args.target_model, args.openai_key)
    target_model = LocalVLLM(args.target_model)
    roberta_model = RoBERTaPredictor('./roberta', device='cuda:0')

    
    questions = []
    questions_str = []
    dataset_path = args.dataset
     
    
    # load question
    with open(dataset_path, newline='') as csvfile:
        reader = csv.reader(csvfile)

        next(reader)
        for row in reader:
            questions.append(row[1])  
            questions_str.append(row[2])  
    
    
    success_list = [0] * len(questions)
    ### pre_jail
    num = 0
    for (question,question_str) in zip(questions,questions_str):
        seed = []
        num += 1
        print("--------QUESTION","pre_jail: ",num,"--------")
        if len(initial_seed) >= 2:
            seed = initial_seed[1:]
        else:
            seed = initial_seed
        fuzzer = GPTFuzzer(
            questions=[question],
            questions_str = [question_str],
            target=target_model,
            predictor=roberta_model,
            initial_seed=seed,
            mutate_policy=MutateRandomSinglePolicy([ 
                OpenAIMutatorscenario(openai_model, temperature=1),
                OpenAIMutatorcharacters(openai_model, temperature=1),
                ],concatentate=False),
            select_policy=MCTSExploreSelectPolicy(),
            energy=args.energy,
            max_jailbreak=args.max_jailbreak,
            max_query=args.pre_query,
            generate_in_batch=False,
            question_id = num
        )



        seed_new = fuzzer.run()
        if seed_new:
            initial_seed.append(seed_new)
            success_list[num-1] = 1
    
    ### final_jail
    num = 0
    for (question,question_str) in zip(questions,questions_str):
        seed = []
        num += 1
        if success_list[num-1] == 1:
            continue
        print("--------QUESTION","final_jail: " , num,"--------")
        if len(initial_seed) >= 2:
            seed = initial_seed[1:]
            mutate_policy=MutateRandomSinglePolicy([ 
                OpenAIMutatorExpand(openai_model, temperature=1),
                OpenAIMutatorscenario(openai_model, temperature=1),
                OpenAIMutatorcharacters(openai_model, temperature=1),
                ],concatentate=False)
        else:
            seed = initial_seed
            mutate_policy=MutateRandomSinglePolicy([ 
                OpenAIMutatorscenario(openai_model, temperature=1),
                OpenAIMutatorcharacters(openai_model, temperature=1),
                ],concatentate=False)
        
        fuzzer = GPTFuzzer(
            questions=[question],
            questions_str = [question_str],
            target=target_model,
            predictor=roberta_model,
            initial_seed=seed,
            mutate_policy=mutate_policy,
            select_policy=MCTSExploreSelectPolicy(),
            energy=args.energy,
            max_jailbreak=args.max_jailbreak,
            max_query=args.max_query-args.pre_query,
            generate_in_batch=False,
            question_id = num
        )
        seed_new = fuzzer.run()
        if seed_new:
            initial_seed.append(seed_new)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fuzzing parameters')
    parser.add_argument('--openai_key', type=str, default='', help='OpenAI API Key')
    parser.add_argument('--claude_key', type=str, default='', help='Claude API Key')
    parser.add_argument('--palm_key', type=str, default='', help='PaLM2 api key')
    parser.add_argument('--model_path', type=str, default='gpt-3.5-turbo',
                        help='mutate model path')
    parser.add_argument('--target_model', type=str, default='',
                        help='The target model, openai model or open-sourced LLMs')
    parser.add_argument('--max_query', type=int, default=100,
                        help='The maximum number of queries')
    parser.add_argument('--max_jailbreak', type=int,
                        default=1, help='The maximum jailbreak number')
    parser.add_argument('--energy', type=int, default=1,
                        help='The energy of the fuzzing process')
    parser.add_argument('--seed_selection_strategy', type=str,
                        default='round_robin', help='The seed selection strategy')
    parser.add_argument("--seed_path", type=str,
                        default="datasets/empty.csv")
    parser.add_argument("--pre_query",type=int, default=10,
                        help='The maximum number of pre queries')
    parser.add_argument("--dataset",type=str, default='./datasets/questions/harmful_behaviors_custom.csv',
                        help='questions to jailbreak')
    add_model_args(parser)

    args = parser.parse_args()
    main(args)
