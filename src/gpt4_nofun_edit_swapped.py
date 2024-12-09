import os
import time
import argparse
import jsonlines
from tqdm import tqdm
import pandas as pd
import litellm

#Input all the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data_subset", type = str, default = "latency", help = "type of non-func requirement")
parser.add_argument("--temperature", type = float, default = 0, help = "temperature")
parser.add_argument("--max_new_tokens", type = int, default = 5192, help = "max length of tokens")
parser.add_argument("--prompt", type = str, default = "base_prompt", help = "type of prompt")
parser.add_argument("--model_path", type = str, default = "gpt-4o", help = "type of prompt")
args = parser.parse_args()
argsdict = vars(args)


def model_query(all_messages, batch_size = 1):

    all_messages = [messages[0] for messages in all_messages]

    start_time = time.time()
    all_generated_answers = []
    for i in tqdm(range(len(all_messages))):
        llm_outputs = litellm.completion(
                api_key="<your_key>",
                base_url="https://cmu.litellm.ai",
                model="openai/gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant. Your response must follow the same format as that of the program given in input by the user. Your output code must be inside ```<programming_language_in_input>```."},
                    all_messages[i]
                ],
                temperature=args.temperature,
                max_completion_tokens=5192,
        )
        all_generated_answers.append([all_messages[i]['content'] + llm_outputs['choices'][0]['message']['content']])
        #print(llm_outputs['choices'][0]['message']['content'])
        
                
    total_time = time.time() - start_time
    avg_times = [total_time / len(all_messages)] * len(all_messages)
    
    return all_generated_answers, avg_times


dataset_path = os.path.join("datasets_swapped", f"{args.data_subset}.jsonl")

data = []
max_tokens = []
generations = []
all_messages = []

with jsonlines.open(dataset_path) as data_file:
    for data_item in data_file:
        data.append(data_item)
        content = data_item[args.prompt]
        messages=[{"role": "user", "content": content}]
        all_messages.append(messages)

print("Starting model inference...")
all_generated_answers, all_inference_times = model_query(all_messages = all_messages)    

for i, data_item in tqdm(enumerate(data)):
    generated_answers = all_generated_answers[i]
    inference_time = all_inference_times[i]
    curr_sample = data_item
    curr_sample["inference_time"] = inference_time
    curr_sample["generated_answers"] = generated_answers

    for prompt in ["base_prompt", "coding_concepts", "chain_of_thought", "one_shot"]:
        del curr_sample[prompt]
    
    generations.append(curr_sample)   
    
generations = pd.DataFrame(generations)
path = os.path.join("generations_swapped", "edit", args.data_subset, os.path.split(args.model_path)[1], args.prompt, "1_samples")
if not os.path.exists(path):
    os.makedirs(path)
path=os.path.join(path, "generated_outputs.jsonl")
generations.to_json(path, orient = "records", lines=True)
