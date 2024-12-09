import os
import pandas as pd
import argparse
import jsonlines
from tqdm import tqdm
import litellm

#Input all the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data_subset", type = str, default = "latency", help = "type of non-func requirement")
parser.add_argument("--temperature", type = float, default = 0.0, help = "temperature")
parser.add_argument("--max_new_tokens", type = int, default = 8, help = "max length of tokens")
parser.add_argument("--top_p", type = float, default = 0.95, help = "top_p")
parser.add_argument("--prompt", type = str, default = "base_prompt", help = "type of prompt")
parser.add_argument("--num_samples", type = int, default = 1, help = "number of samples")
parser.add_argument("--model_path", type = str, required = True, help = "HF path for OS models")
parser.add_argument("--load_in_8bit", action = "store_true", help = "Load model in 8bit")
parser.add_argument("--load_in_4bit", action = "store_true", help = "Load model in 4bit")
parser.add_argument("--precision", type = str, default = "fp16", help = "Model precision, from: fp32, fp16 or bf16")
parser.add_argument("--tensor_parallel_size", type = int, default = 1, help = "Tensor parallel size")
parser.add_argument("--swap_space", type = int, default = 12, help = "The size (GiB) of CPU memory per GPU to use as swap space.")
parser.add_argument("--batch_size", type = int, default = 1, help = "Number of examples to send to llm engine at once.")
args = parser.parse_args()
argsdict = vars(args)
 
# Function to extract the classification prediction 
def extract_single_predictions(input_string):
    if input_string.strip().split()[0].lower() == "A".lower():
        return "A"
    elif input_string.strip().split()[0].lower() == "B".lower():
        return "B"
    return None
 
model_basename = args.model_path.split("/")[-1]
 
GREEDY = True

# Initializing variables 
dataset_path = os.path.join("datasets", f"{args.data_subset}.jsonl") 
args.num_samples = 1
data = []
max_tokens = []
generations = []
left_prompts = []
right_prompts = []
generations = []
def model_query(all_messages, batch_size = 1):
    all_generated_answers = []
    for i in tqdm(range(len(all_messages))):
        llm_outputs = litellm.completion(
                api_key="<your_key>",
                base_url="https://cmu.litellm.ai",
                model="openai/gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant. Your output must be exactly one character as either 'A' or 'B' depending on which code satisfies the non-functional requirements mentioned by the user."},
                    {"role": "user", "content": all_messages[i]},
                ],
                temperature=0.0,
                max_completion_tokens=4,
        )
        all_generated_answers.append(llm_outputs['choices'][0]['message']['content'])
        # print(all_generated_answers[-1])
                
    # total_time = time.time() - start_time
    # avg_times = [total_time / len(all_messages)] * len(all_messages)
    
    return all_generated_answers 

with jsonlines.open(dataset_path) as data_file:    
    for data_item in data_file:
        data.append(data_item)
        left_prompts.append(data_item["classification_left_prompt"])
        right_prompts.append(data_item["classification_right_prompt"])
 
print("Starting model inference...")

left_llm_outputs = model_query(left_prompts, 1)
left_predictions = [extract_single_predictions(output) for output in left_llm_outputs]
right_llm_outputs = model_query(right_prompts, 1)
right_predictions = [extract_single_predictions(output) for output in right_llm_outputs]
 
for i, data_item in tqdm(enumerate(left_predictions)):

    curr_sample = data[i]
    curr_sample["left_output"] = left_predictions[i]
    curr_sample["right_output"] = right_predictions[i]

    for prompt in ["base_prompt", "coding_concepts", "chain_of_thought", "one_shot", "classification_left_prompt", "classification_right_prompt"]:

        if(prompt in curr_sample):
            del curr_sample[prompt]

    generations.append(curr_sample)  

# Saving the generations   
generations = pd.DataFrame(generations)
path = os.path.join("generations", "classification", args.data_subset, os.path.split(args.model_path)[1], args.prompt, f"{args.num_samples}_samples")

if not os.path.exists(path):
    os.makedirs(path)

path = os.path.join(path, "generated_outputs.jsonl")
generations.to_json(path, orient = "records", lines = True)
