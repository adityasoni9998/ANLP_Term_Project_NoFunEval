import json

with open('generated_outputs.jsonl', 'r') as f:
  lines = f.readlines()
  op = []
  for l in lines:
    d = json.loads(l.strip())
    pl = d['pl'].lower()
    gen = d['generated_answers'][0]
    gen = gen.replace(f'```{pl}\n```{pl}', f'```{pl}\n')
    d['generated_answers'] = [gen]
    op.append(d)

with open('generated_outputs.jsonl', 'w') as f:
  for o in op:
    f.write(json.dumps(o) + '\n')

