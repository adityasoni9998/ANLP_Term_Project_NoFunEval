import argparse
import os 
import sys
import tempfile
import jsonlines
import subprocess
from utils import remove_blank_lines, remove_comments, post_process_generations
import tokenize
from io import StringIO
import editdistance
from nltk.translate.bleu_score import sentence_bleu
from codebleu import calc_codebleu
import code_bert_score


def count_diff_lines(diff_output):
    """Count number of added and removed lines in git diff output"""
    added = len([l for l in diff_output.splitlines() if l.startswith('+')])
    removed = len([l for l in diff_output.splitlines() if l.startswith('-')])
    return added, removed

def get_token_distance(code1, code2):
    """Calculate token-level edit distance between two code snippets"""
    def tokenize_code(code):
        tokens = []
        try:
            for tok in tokenize.generate_tokens(StringIO(code).readline):
                if tok.type not in [tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE]:
                    tokens.append(tok.string)
        except:
            tokens = code.split()
        return tokens
    
    tokens1 = tokenize_code(code1)
    tokens2 = tokenize_code(code2)
    return Levenshtein.distance(' '.join(tokens1), ' '.join(tokens2))

def calculate_edit_metrics(source_code, generated_code, pl):
    """Calculate various edit distance metrics between source and target code"""
    source = remove_blank_lines(remove_comments(source_code, pl.lower()))
    target = remove_blank_lines(remove_comments(generated_code, pl.lower()))
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as source_temp, \
         tempfile.NamedTemporaryFile(mode='w', delete=False) as target_temp:
        
        source_temp.write(source)
        target_temp.write(target)
        source_path = source_temp.name
        target_path = target_temp.name

    command_diff = "git diff -U0 --no-index --ignore-all-space --ignore-blank-lines {} {} | tail -n +5 | grep -v 'No newline at end of file'".format(
        source_path, target_path)
    
    diff_output = subprocess.run(command_diff, shell=True, stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE).stdout.decode()
    
    metrics = {
        'git_diff': count_diff_lines(diff_output)[0] + count_diff_lines(diff_output)[1],

        'edit_distance': editdistance.eval(source, target)
    }
    
    # Clean up temp files
    os.unlink(source_path)
    os.unlink(target_path)
    
    return metrics


def print_diff(source_code, target, pl):

    with tempfile.NamedTemporaryFile(mode = 'w', delete = False) as source_temp, tempfile.NamedTemporaryFile(mode = 'w', delete = False) as target_temp, tempfile.NamedTemporaryFile(mode = 'w', delete = False) as generated_temp:

        source_temp.write(remove_blank_lines(remove_comments(source_code,pl.lower())))
        target_temp.write(remove_blank_lines(remove_comments(target,pl.lower())))
        
        source_path = source_temp.name
        target_path = target_temp.name

    command_diff_target = "git diff -U0 --no-index --ignore-all-space --ignore-blank-lines {} {} | tail -n +5 | grep -v 'No newline at end of file'".format(source_path,target_path)
    
    diff_target = subprocess.run(command_diff_target, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.decode()

    print(diff_target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_subset", type = str, default = "resource_util", help = "latency/resource_util/runtime_efficiency/maintenance/security")
    parser.add_argument("--model", type = str, default = "CodeLlama-7b-Instruct-hf", help = "model name")
    parser.add_argument("--prompt", type = str, default = "coding_concepts", help = "base_prompt/coding_concepts/chain_of_thought/one_shot")
    parser.add_argument("--num_samples", type = int, default = 1, help = "Number of samples")
    parser.add_argument("--score_k", type = str, default = "1,5,10,20", help = "K value for score@k (should not be greater than num_samples and can be comma-separated)")
    parser.add_argument("--metric", type = str, default = "code_bert_score", help = "[git_diff, edit_distance, bleu, codebleu, bert_score, code_bert_score]")
    args = parser.parse_args()
    
    generations_path_original = os.path.join("/home/srgandhi/NoFunEval/generations_original", "edit", args.data_subset, args.model, args.prompt, f"{args.num_samples}_samples", "generated_outputs.jsonl")
    generations_path_swapped = os.path.join("/home/srgandhi/NoFunEval/generations_swapped", "edit", args.data_subset, args.model, args.prompt, f"{args.num_samples}_samples", "generated_outputs.jsonl")
    results_path = os.path.join("/home/srgandhi/NoFunEval/results", "classification_as_edit", args.metric, args.data_subset, args.model, args.prompt, f"{args.num_samples}_samples")
    
    # if not (os.path.isdir(os.path.join("/home/srgandhi/NoFunEval/generations_original", "edit", args.data_subset, args.model, args.prompt, f"{args.num_samples}_samples"))) and os.path.isdir(os.path.join("/home/srgandhi/NoFunEval/generations_swapped", "edit", args.data_subset, args.model, args.prompt, f"{args.num_samples}_samples")):
    #     sys.exit()

    metrics_original_all, metrics_swapped_all = [], []
    source_original_all, target_swapped_all, generation_original_all, generation_swapped_all = [], [], [], []
    unsupported_langs = set()

    with jsonlines.open(generations_path_original) as reader:

        for generation in reader:

            for l in range(args.num_samples):
                
                generated_answers_original = post_process_generations(generated_answers = generation['generated_answers'][l], model = args.model, prompt = args.prompt, pl = generation['pl'])
                
                if args.metric == 'edit_distance' or args.metric == 'git_diff':
                    metrics_original = calculate_edit_metrics(source_code = generation['source_code'], generated_code = generated_answers_original[1], pl = generation['pl'])
                
                elif args.metric == 'codebleu':
                    # generation['source_code'] = remove_blank_lines(remove_comments(generation['source_code'], generation['pl'].lower()))
                    # generated_answers_original[1] = remove_blank_lines(remove_comments(generated_answers_original[1], generation['pl'].lower()))
                    try:
                        metrics_original = calc_codebleu([generation['source_code']], [generated_answers_original[1]], lang = generation['pl'].lower())
                    except AssertionError as e:
                        if 'is not supported' in str(e):
                            unsupported_langs.add(generation['pl'].lower())
                            metrics_original = {}
                            metrics_original['codebleu'] = sentence_bleu([generation['source_code']], generated_answers_original[1])
                        else:
                            print('Error is not about unsupported language')
                            sys.exit()
                
                elif args.metric == 'bleu':
                    # generation['source_code'] = remove_blank_lines(remove_comments(generation['source_code'], generation['pl'].lower()))
                    # generated_answers_original[1] = remove_blank_lines(remove_comments(generated_answers_original[1], generation['pl'].lower()))
                    metrics_original = {}
                    metrics_original['bleu'] = sentence_bleu([generation['source_code']], generated_answers_original[1])
                
                elif args.metric == 'code_bert_score':
                    metrics_original = {}
                    metrics_original['code_bert_score'] = float(code_bert_score.score([generation['source_code']], [generated_answers_original[1]], lang = generation['pl'].lower())[2][0])
                    
                source_original_all.append(generation['source_code'])
                generation_original_all.append(generated_answers_original[1])

                # print(metrics_original)
                metrics_original_all.append(metrics_original)

    # print('------------Swapped-------')   
    with jsonlines.open(generations_path_swapped) as reader:

        for generation in reader:

            for l in range(args.num_samples):
                
                generated_answers_swapped = post_process_generations(generated_answers = generation['generated_answers'][l], model = args.model, prompt = args.prompt, pl = generation['pl'])
                
                if args.metric == 'edit_distance' or args.metric == 'git_diff':
                    metrics_swapped = calculate_edit_metrics(source_code = generation['target_code'], generated_code = generated_answers_swapped[1], pl = generation['pl'])
                
                elif args.metric == 'codebleu':
                    # generation['target_code'] = remove_blank_lines(remove_comments(generation['target_code'], generation['pl'].lower()))
                    # generated_answers_swapped[1] = remove_blank_lines(remove_comments(generated_answers_swapped[1], generation['pl'].lower()))
                    try:
                        metrics_swapped = calc_codebleu([generation['target_code']], [generated_answers_swapped[1]], lang = generation['pl'].lower())
                    except AssertionError as e:
                        if 'is not supported' in str(e):
                            unsupported_langs.add(generation['pl'].lower())
                            metrics_swapped = {}
                            metrics_swapped['codebleu'] = sentence_bleu([generation['target_code']], generated_answers_swapped[1])
                        else:
                            print('Error is not about unsupported language')
                            sys.exit()
                
                elif args.metric == 'bleu':
                    # generation['target_code'] = remove_blank_lines(remove_comments(generation['target_code'], generation['pl'].lower()))
                    # generated_answers_swapped[1] = remove_blank_lines(remove_comments(generated_answers_swapped[1], generation['pl'].lower()))
                    metrics_swapped = {}
                    metrics_swapped['bleu'] = sentence_bleu([generation['target_code']], generated_answers_swapped[1])
                
                elif args.metric == 'code_bert_score':
                    metrics_swapped = {}
                    metrics_swapped['code_bert_score'] = float(code_bert_score.score([generation['target_code']], [generated_answers_swapped[1]], lang = generation['pl'].lower())[2][0])
                    

                target_swapped_all.append(generation['target_code'])
                generation_swapped_all.append(generated_answers_swapped[1])
                
                # print(metrics_swapped)
                metrics_swapped_all.append(metrics_swapped)
    # sys.exit()
    predictions = []
    
    # count = 0
    # zeros = 0
    # equals = 0
    # for i in range(len(generation_original_all)):
    #     assert (source_original_all[i] != target_swapped_all[i])
    #     if metrics_swapped_all[i][args.metric] == metrics_original_all[i][args.metric] and metrics_swapped_all[i][args.metric] == 0:
    #         equals += 1
    #     if generation_original_all[i] == generation_swapped_all[i]:
    #         count += 1
    #         if metrics_swapped_all[i][args.metric] == 0 and metrics_original_all[i][args.metric] == 0:
    #             zeros += 1
        

    # print(count/len(generation_original_all))
    # print(zeros/len(generation_original_all))
    # print(equals/len(generation_original_all))
    # sys.exit()

    for i in range(len(metrics_swapped_all)):

        if args.metric == 'edit_distance' or args.metric == 'git_diff':
            if metrics_swapped_all[i][args.metric] < metrics_original_all[i][args.metric]:
                predictions.append(1)
            else:
                predictions.append(0)
        elif args.metric in ['bleu', 'codebleu', 'bert_score', 'code_bert_score']:
            if metrics_swapped_all[i][args.metric] > metrics_original_all[i][args.metric]:
                predictions.append(1)
            else:
                predictions.append(0)

    print(predictions)
    print('Classification Accuracy: ', sum(predictions)/len(predictions))
    print(f'Unsupported Languages for metric {args.metric}: ', unsupported_langs)
    # sys.exit()
    os.makedirs(results_path, exist_ok=True)

    with open(f'{results_path}/accuracy.txt', 'w') as f:
        f.write(str(100 * sum(predictions)/len(predictions)))
    
    with open(f'{results_path}/preds.txt', 'w') as f:
        f.writelines([str(f'{pred}\n') for pred in predictions])
                