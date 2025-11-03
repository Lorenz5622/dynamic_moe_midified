from transformers import AutoTokenizer
import torch
from Dynamic_MoE.modeling.modeling_moe import MoEForCausalLM
from Dynamic_MoE.modeling.configuration_moe import MoEConfig
import json
import torch.nn.functional as F
import random
from Dynamic_MoE.rl.rl import GeneticAlgorithm
import numpy as np
import os
from datetime import datetime
N_GENERATIONS = 500
CROSSOVER_RATE = 0.6
MUTATION_RATE = 0.08
MAX_DATASET_COUNT = 150
MAX_DATASET_EPOCHS = 40
INDIVIDUAL_COUNT = 30
PATH_PREFIX = "/root"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def generate(tokenizer, model, text, dynamic_k=None):
    inputs = [text]
    tokens = tokenizer(inputs,return_tensors="pt")
    input_ids = tokens.input_ids.cuda()
    generate_ids = model.generate(inputs=input_ids,
                num_beams=1, 
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                dynamic_k=dynamic_k,
                max_new_tokens=1,top_p=0.9, temperature=1.0, do_sample=True)
                # max_new_tokens=32, do_sample=False)
    outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    response = [outputs[i][len(inputs[i]):] for i in range(len(outputs))][0]
    response_ids = generate_ids[:, input_ids.shape[1]:]
    return response, response_ids

def generate_batch(tokenizer, model, prompts, dynamic_k=None):
    try:
        with torch.no_grad():
            tokens = tokenizer(prompts, return_tensors="pt", padding=True)
            input_ids = tokens.input_ids.cuda()
            
            generate_ids = model.generate(
                inputs=input_ids,
                num_beams=1, 
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                dynamic_k=dynamic_k,
                max_new_tokens=1,
                top_p=0.9, 
                temperature=1.0, 
                do_sample=True
            )
            
            outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
            # 提取生成部分（去除输入部分）
            responses = []
            response_ids = []
            for i, output in enumerate(outputs):
                input_length = input_ids.shape[1]  # 统一的输入长度
                response = output[input_length:]  # 跳过输入部分
                responses.append(response)
                response_ids.append(generate_ids[i, input_length:])  # 提取生成的token IDs
            
            return responses, torch.stack(response_ids)
    except torch.cuda.OutOfMemoryError:
        # print("CUDA out of memory. Clearing cache and returning empty results.")
        torch.cuda.empty_cache()
        return [], torch.empty(0)  # 返回空结果

def calculate_batch_loss(tokenizer, model, prompts, labels, isOOM):
    if isOOM:
        # print(f"ce_loss.mean(): 1000")
        return 1000
    try:
        with torch.no_grad():
            # 获取模型输出
            outputs = model.saved_logits[0]  # [batch_size, seq_len, vocab_size]
            
            # 批量tokenize输入
            inputs = tokenizer(prompts, return_tensors="pt", padding=True)
            input_ids = inputs.input_ids.cuda()
            
            # 构造 shift_logits 和 shift_labels
            shift_logits = outputs[..., :-1, :].contiguous().float()
            shift_labels = input_ids[..., 1:].contiguous()
            
            # 计算 loss
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=tokenizer.pad_token_id)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1)).view(shift_labels.size())
            
            # 计算有效长度
            lens = (input_ids != tokenizer.pad_token_id).sum(-1).cpu().numpy()
            
            # 计算平均 loss
            ce_loss = loss.float().sum(-1).cpu().detach().numpy() / lens
            # print(f"ce_loss.mean(): {ce_loss.mean()}")
            return ce_loss.mean()
    except torch.cuda.OutOfMemoryError:
        # print("CUDA out of memory in calculate_batch_loss. Clearing cache and returning empty results.")
        torch.cuda.empty_cache()
        return 1000
    
def find_pattern(output_ids):
    # 定义目标模式
    pattern_1 = [673, 29901]
    candidates = [29909, 29933, 29907]

    # 转换为 tensor 方便处理
    output_ids = output_ids.view(-1)  # 确保是一维

    found_positions = []

    for i in range(len(output_ids) - len(pattern_1)):
        # 检查是否匹配前两个 token: [673, 29901]
        if output_ids[i:i+2].tolist() == pattern_1:
            # 检查下一个 token 是否是候选之一
            next_token = output_ids[i+2].item()
            if next_token in candidates:
                found_positions.append(i+2)  # 记录位置和具体哪个 token
                break

    return found_positions

def analysis_siqa(line_json, line_label):
    data = json.loads(line_json.strip())
    prompt = f"""
        Context: {data['context']}
        Question: {data['question']}
        Options:
        A: {data['answerA']}
        B: {data['answerB']}
        C: {data['answerC']}
        Answer:"""
    return prompt

def analysis_piqa(line_json, line_label):
    data = json.loads(line_json.strip())
    prompt = f"""
        Question: {data['goal']}
        Options:
        A: {data['sol1']}
        B: {data['sol2']}
        Answer:"""
    return prompt

def random_n_dataset(file_json, file_label):
    output_data = f'{PATH_PREFIX}/datasets/piqa/sampled_{MAX_DATASET_COUNT*MAX_DATASET_EPOCHS}_data.jsonl'
    output_label = f'{PATH_PREFIX}/datasets/piqa/sampled_{MAX_DATASET_COUNT*MAX_DATASET_EPOCHS}_labels.lst'

    # 读取数据和标签，并构建成 pairs
    with open(file_json, 'r', encoding='utf-8') as f_json, \
        open(file_label, 'r', encoding='utf-8') as f_label:

        pairs = [
            (json.loads(data_line.strip()), label_line.strip())
            for data_line, label_line in zip(f_json, f_label)
            if data_line.strip() and label_line.strip()
        ]
    sampled_pairs = random.sample(pairs, MAX_DATASET_COUNT*MAX_DATASET_EPOCHS)

    with open(output_data, 'w', encoding='utf-8') as f_data, \
        open(output_label, 'w', encoding='utf-8') as f_label:

        for data, label in sampled_pairs:
            # 保存原始数据（不加 label 字段）
            f_data.write(json.dumps(data, ensure_ascii=False) + '\n')
            # 保存标签（每行一个）
            f_label.write(label + '\n')
    
    return output_data, output_label

if __name__ == "__main__":
    if os.path.exists("/home/cyx") :
        PATH_PREFIX = "/home/cyx"
    model_path = f'{PATH_PREFIX}/models/Dynamic_MoE'
    now = datetime.now().strftime("%Y%m%d_%H%M%S")  # 例如：20251007_213015

    # 构造文件名
    record_file = f"records/output_{now}.txt"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.unk_token

    model_config = MoEConfig.from_pretrained(model_path,trust_remote_code=True)
    model = MoEForCausalLM.from_pretrained(
        model_path,
        from_tf=False,
        config=model_config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).cuda()
    model.eval()
    model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
    hidden_layers = 0
    with open(f'{PATH_PREFIX}/models/Dynamic_MoE/config.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
        hidden_layers = data['num_hidden_layers']

    file_json = f'{PATH_PREFIX}/datasets/piqa/train.jsonl'   # 第一个文件路径
    file_label = f'{PATH_PREFIX}/datasets/piqa/train-labels.lst'  # 第二个文件路径
    file_json, file_label = random_n_dataset(file_json, file_label)

    print(file_json, file_label,)
    # 抽取N条保存为另一个文件中

    # ga = GeneticAlgorithm(60, 32*3) # 24代表有24层，3代表每一层可以从000-111（二进制转化后为0-7）个专家中选择
    ga = GeneticAlgorithm(INDIVIDUAL_COUNT, 24) # 24代表有24层
    pop = ga.pop
    print(f"--------------begin iterations at {now} ---------------------")
    for gen_count in range(N_GENERATIONS):  # 迭代N代
        expert_list = ga.translateDNA()
        loss_list = []
        oomCount = 0
        
        with open(file_json, 'r', encoding='utf-8') as fj, \
            open(file_label, 'r', encoding='utf-8') as fl:
            
            data_pairs = list(zip(fj.readlines(), fl.readlines()))
            # 打乱顺序
            random.shuffle(data_pairs)
            
            # 限制处理的数据量
            data_pairs = data_pairs[:MAX_DATASET_COUNT*MAX_DATASET_EPOCHS]
            epochs_data = []
            for epoch in range(MAX_DATASET_EPOCHS):
                start_idx = epoch * MAX_DATASET_COUNT
                end_idx = start_idx + MAX_DATASET_COUNT
                epoch_data = data_pairs[start_idx:end_idx]
                epochs_data.append(epoch_data)
            
            for i, experts in enumerate(expert_list):
                now1 = datetime.now().strftime("%Y%m%d_%H%M%S")
                print(f"expert {i} begin training, at time {now1}")
                # 批量处理所有数据
                epoch_losses = []

                for epoch_data in epochs_data:
                    prompts = []
                    labels = []

                    for line_json, line_label in epoch_data:
                        prompt = analysis_piqa(line_json, line_label)
                        prompts.append(prompt)
                        labels.append(line_label.strip())
                    
                    # 批量生成
                    responses, output_ids = generate_batch(tokenizer, model, prompts, experts.tolist())
                    
                    # 批量计算loss
                    isOOM = False
                    if len(responses) == 0:
                        isOOM = True
                        oomCount += 1
                    batch_loss = calculate_batch_loss(tokenizer, model, prompts, labels, isOOM)
                    if batch_loss < 1000:
                        epoch_losses.append(batch_loss)
                    
                    # 清理缓存
                    model.saved_logits = []
                    model.collected_hidden_states = []
                    # 解决3次interation之后内存不够的问题
                    if hasattr(model, 'past_key_values'):
                        del model.past_key_values
                    torch.cuda.empty_cache()

                # print(f"epoch_losses: {epoch_losses}")
                if len(epoch_losses) == 0:
                    loss_list.append(1000)
                else:
                    loss_list.append(np.mean(epoch_losses))
        
        if oomCount > INDIVIDUAL_COUNT*MAX_DATASET_EPOCHS/4:
            with open(record_file, 'a') as f:
                f.write(f"jump this interation {gen_count} due to frequant CUDA OOM\n")
            print("jump this interation due to frequant CUDA OOM")
            continue
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"generation No. {gen_count}, time {now}: ")
        fitness = ga.get_fitness(np.array(loss_list))
        ga.print_info()
        if gen_count % 2 == 0:
            ga.write_to_record(record_file, gen_count)
        pop = ga.select(
            elite_rate = 0.1,
            diversity_threshold = 0.1,
            cross_rate = CROSSOVER_RATE,
            mutation_rate = MUTATION_RATE
        )  # 选择生成新的种群


        
            


    
    
    
