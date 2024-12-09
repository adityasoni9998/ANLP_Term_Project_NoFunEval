import json

data_subsets = ['latency', 'maintainability', 'runtime_efficiency', 'security', 'resource_util']
prompt_types = ['chain_of_thought', 'one_shot', 'base_prompt', 'coding_concepts']
cnt = 0

for data_subset in data_subsets:
	filename = f"./datasets/{data_subset}.jsonl"
	new_file = f"./datasets_swapped/{data_subset}.jsonl"
	orig_data_list = []
	new_data_list = []
	with open(new_file, 'r') as f:
		lines = f.readlines()
		for line in lines:
			new_data_list.append(json.loads(line.strip()))
	for data in new_data_list:
		source_code = data['source_code']
		target_code = data['target_code']
		for prompt in prompt_types:
			assert target_code in data[prompt]
			if source_code not in target_code:
				assert source_code not in data[prompt]
	