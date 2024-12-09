# code to create the swapped dataset

import json

data_subsets = ['latency', 'maintainability', 'runtime_efficiency', 'security', 'resource_util']
prompt_types = ['chain_of_thought', 'one_shot', 'base_prompt', 'coding_concepts']
cnt = 0

for data_subset in data_subsets:
	filename = f"./datasets/{data_subset}.jsonl"
	new_file = f"./datasets_swapped/{data_subset}.jsonl"
	new_data_list = []
	with open(filename, 'r') as f:
		lines = f.readlines()
		for line in lines:
			data = json.loads(line.strip())
			new_data_point = data
			source_code = data['source_code']
			target_code = data['target_code']
			
			for prompt in prompt_types:
				assert source_code in data[prompt]
			for prompt in prompt_types:
				new_data_point[prompt] = new_data_point[prompt].replace(source_code, target_code)
			new_data_list.append(new_data_point)
			cnt += 1
	with open(new_file, 'w') as f:
		for data in new_data_list:
			f.write(json.dumps(data) + '\n')
print(cnt)