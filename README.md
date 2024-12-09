# Introduction
This is the code and dataset used for the ANLP term project and is based on the paper ["NoFunEval: Funny How Code LMs Falter on Requirements Beyond Functional Correctness"](https://arxiv.org/abs/2401.15963). The original code source is [here](https://github.com/microsoft/NoFunEval/tree/main).

## Repository Contents
1. [Original Datasets](./datasets): The original NoFunEval Dataset.
2. [Swapped Datasets](./datasets_swapped): Modified dataset where we replace the source code by the target code in input.
2. [Evaluation scripts](#3-evaluation-scripts): Scripts to evaluate LMs by taking input from examples in NoFunEval and producing score@k scores for the metrics reported in the paper: DiffBleu, Average SpeedUp, CodeQL, CodeQL-DiffBleu.

## Datasets
* **Runtime Efficiency**:
Improving algorithmic time complexity of code for faster run-time.
* **Maintainability**: 
Improving code readability and style, and following the best programming practices.
* **Latency**:
Optimizing code for faster response times in Android applications.
* **Resource Utilization**:
Optimizing code for lower resource utilization on edge devices.
* **Security**:
Fixing specific security vulnerabilities in the input code.
* **HumanEvalClassify**:
Classifying between the buggy and correct snippets of code. 

# Environment Setup
Create a conda environment and activate it (Python version must be 3.8, we face errors during installation for more recent versions). 
```console
conda create -n nofuneval python=3.8
conda activate nofuneval
```

Install all the requirements (except vllm):
```console
bash setup.sh
```

Install vllm at the end:
```console
pip install vllm
```

# Generation
### Original dataset
For all models from HuggingFace we use the below command:
```console
python3 src/nofunedit_generation.py --data_subset <subset from nofunedit: eg-latency> --model_path <model name from HF: eg-WizardLM/WizardCoder-15B-V1.0> --temperature <temperature to be set for model generation: eg-0> --max_new_tokens <maximum number of new tokens to be generated: eg-5192> --prompt <type of prompt to use from our dataset: eg-base_prompt> --num_samples <number of samples to be generated: eg-1> --precision <floating point format: eg-fp16> --batch_size <number of examples to send to llm engine at once: eg-1>
```
For GPT-4o, we use the below command:
```console
python3 src/gpt4_nofun_edit.py --data_subset <subset from nofunedit: eg-latency> --temperature <temperature to be set for model generation: eg-0> --max_new_tokens <maximum number of new tokens to be generated: eg-5192> --prompt <type of prompt to use from our dataset: eg-base_prompt>
```
### Swapped Dataset
For all models from HuggingFace we use the below command:
```console
python3 src/nofunedit_generation_swapped.py --data_subset <subset from nofunedit: eg-latency> --model_path <model name from HF: eg-WizardLM/WizardCoder-15B-V1.0> --temperature <temperature to be set for model generation: eg-0> --max_new_tokens <maximum number of new tokens to be generated: eg-5192> --prompt <type of prompt to use from our dataset: eg-base_prompt> --num_samples <number of samples to be generated: eg-1> --precision <floating point format: eg-fp16> --batch_size <number of examples to send to llm engine at once: eg-1>
```

For GPT-4o, we use the below command:
```console
python3 src/gpt4_nofun_edit_swapped.py --data_subset <subset from nofunedit: eg-latency> --temperature <temperature to be set for model generation: eg-0> --max_new_tokens <maximum number of new tokens to be generated: eg-5192> --prompt <type of prompt to use from our dataset: eg-base_prompt>
```

### Classification
For open-source models on HuggingFace, run:
```console
python3 src/classification_generation.py --data_subset <subset from non_func or humanevalclassify: eg-latency> --model <model name from HF: eg-WizardLM/WizardCoder-15B-V1.0> --temperature <temperature to be set for model generation: eg-0> --max_new_tokens <maximum number of new tokens to be generated: eg-5192> --prompt <type of prompt to use from our dataset: eg-base_prompt> --precision <floating point format: eg-fp16> --batch_size <number of examples to send to llm engine at once: eg-1>
```

For GPT-4o, run:
```console
python3 src/gpt4_nofun_classify.py --data_subset <subset from non_func or humanevalclassify: eg-latency> --temperature <temperature to be set for model generation: eg-0> --max_new_tokens <maximum number of new tokens to be generated: eg-4> --prompt <type of prompt to use from our dataset: eg-base_prompt>
```
# Evaluation Scripts

## Evaluation
```console
python3 src/evaluation.py --data_subset <subset from nofunedit: eg-latency> --model_path <model name from HF: eg-WizardLM/WizardCoder-15B-V1.0> --prompt <type of prompt to use from our dataset: eg-base_prompt> --num_samples <number of samples to be generated: eg-1> --score_k <K values for score@k: eg-1,5,10,20> --metric <eval_metric to be used: eg-diffbleu>
```

### Example eval script
```console
bash evaluation_example_script.sh
```
For maintainability and security subsets, first run src/evaluation.py for diffbleu and codeql metrics. After this, run the src/evaluation.py code to obtain the codeql-diffbleu metric.
## Parameters

| Parameter                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `data_subset`                     | The subset of data to use. Options: `latency`, `resource_util`, `maintainability`, `security`, `runtime_efficiency` for nofunedit. Additionally `humanevalclassify` for classification.|
| `model_path` | The path of the model from HF. Example: `WizardLM/WizardCoder-15B-V1.0`.
| `prompt`      | Prompt to use. Options: `base_prompt`, `one_shot`, `chain_of_thought`, `coding_concepts`. |
| `num_samples` | Number of samples to generate. Example: `1` (We used  `1` for greedy and `20` for higher temperature). **[NoFunEdit - Generation only]**|
| `max_new_tokens` | Budget for new token generation for a model. Example: `1200` (NoFunEdit: We used `1200` for runtime_efficiency and security for all prompts other than CoT where `1500` was used. For others, we used `5192` or max possible limit. Classification: We used `4` for all generations).|
| `temperature` | Temperature for model generation. Example: `0` (We used `0` for greedy and `0.8` for 20 samples) |
| `score_k` |K vales for Score@K. Example: `1,5,10,20` (Should not be greater than num_samples and is comma separated, authors always report scores for K=1)  **[Eval only]** |
| `metric` | Metric to be used for evaluation. Option:  `diffbleu`, `codeql`, `codeql-diffbleu` (to be run after first two params are run), `classification`, `runtime` **[Eval only]**|

#### VLLM Parameters (for generation)
| Parameter                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `batch-size` | Batch size. Default: `1`|
| `precision` | Floating point format: Default: `fp16` |
|  `tensor_parallel_size` | Default: `1` |
| `swap_space` | The size (GiB) of CPU memory per GPU to use as swap space: Default: `12` |

## Qualitative Examples
The `error_analysis` directory contains full text of the examples mentioned in Appendix A of our report.

## GPU Hardware
All our reproducibility experiments have been run using AWS g6e.xlarge instance which consists of a single NVIDIA L40S GPU with ~46GB memory. For GPT-4o, we query LiteLLM proxy using the API credits for this semester.

## Model outputs
All the model outputs from original dataset are provided in [generations_original](./generations_original) and all the model outputs from the swapped dataset are provided in [generations_swapped](./generations_swapped).

## Code Reference
The evaluation code for runtime efficiency has been derived from the [PIE codebase](https://github.com/madaan/pie-perf).
