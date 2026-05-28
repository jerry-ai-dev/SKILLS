# Lesson 9: 通用模式提炼 & 代码模板

## 学习目标
- 从三个项目（TRL/Open-R1/SimpleRL-Zoo）中提炼共同代码模式
- 整理可复用的 SFT、GRPO、Reward、评估代码模板
- 形成自己项目的"代码工具箱"

## 知识小节

### 小节 1：三个项目的共同模式

所有 SFT+GRPO 项目都遵循同一个模式：

```
1. 加载模型 + tokenizer
2. 准备数据（格式化 + chat template）
3. SFT 阶段：model + labeled_data → SFT model
4. GRPO 阶段：SFT_model + prompts + reward_func → RL model
5. 评估：model + test_data → accuracy
```

### 小节 2：SFT 代码模板

```python
# ===== SFT 模板 =====
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

def run_sft(
    model_name: str,
    dataset_name: str,
    output_dir: str,
    use_lora: bool = True,
    num_epochs: int = 2,
    batch_size: int = 4,
    grad_accum: int = 4,
    lr: float = 2e-5,
    max_seq_length: int = 2048,
):
    # 加载
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="bfloat16")
    dataset = load_dataset(dataset_name)
    
    # LoRA 配置（可选）
    peft_config = LoraConfig(r=64, lora_alpha=16,
                              target_modules=["q_proj","k_proj","v_proj","o_proj"]) if use_lora else None
    
    # 训练配置
    args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        max_seq_length=max_seq_length,
        bf16=True,
        gradient_checkpointing=True,
    )
    
    # 训练
    trainer = SFTTrainer(model=model, args=args, train_dataset=dataset["train"],
                          tokenizer=tokenizer, peft_config=peft_config)
    trainer.train()
    trainer.save_model()
```

### 小节 3：GRPO 代码模板

```python
# ===== GRPO 模板 =====
from trl import GRPOTrainer, GRPOConfig

def run_grpo(
    model_name: str,          # SFT 输出的模型路径
    dataset_name: str,
    reward_funcs: list,
    output_dir: str,
    G: int = 8,
    beta: float = 0.04,
    temperature: float = 0.7,
    lr: float = 1e-6,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="bfloat16")
    dataset = load_dataset(dataset_name)
    
    args = GRPOConfig(
        output_dir=output_dir,
        num_generations=G,
        temperature=temperature,
        beta=beta,
        max_completion_length=512,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=lr,
        bf16=True,
        gradient_checkpointing=True,
    )
    
    trainer = GRPOTrainer(model=model, reward_funcs=reward_funcs,
                           args=args, train_dataset=dataset["train"], tokenizer=tokenizer)
    trainer.train()
    trainer.save_model()
```

### 小节 4：Reward 函数模板

```python
# ===== Reward 模板 =====
import re
import math

def make_accuracy_reward(extract_fn=None):
    """工厂函数：创建准确率奖励"""
    if extract_fn is None:
        extract_fn = extract_last_number
    
    def reward_fn(completions, answer=None, **kwargs):
        rewards = []
        for comp, gt in zip(completions, answer):
            pred = extract_fn(comp)
            if pred is not None and math.isclose(float(pred), float(gt), rel_tol=1e-5):
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        return rewards
    return reward_fn

def make_format_reward(required_tags=None):
    """工厂函数：创建格式奖励"""
    if required_tags is None:
        required_tags = [("<think>", "</think>"), ("<answer>", "</answer>")]
    
    def reward_fn(completions, **kwargs):
        rewards = []
        for comp in completions:
            score = sum(1 for open_tag, close_tag in required_tags
                       if open_tag in comp and close_tag in comp)
            rewards.append(score / len(required_tags))
        return rewards
    return reward_fn

def extract_last_number(text):
    """通用答案提取"""
    # 优先 <answer> 标签
    match = re.search(r'<answer>\s*(-?\d+\.?\d*)\s*</answer>', text)
    if match: return float(match.group(1))
    # 兜底：最后一个数字
    numbers = re.findall(r'-?\d+\.?\d*', text)
    return float(numbers[-1]) if numbers else None
```

### 小节 5：评估模板

```python
# ===== 评估模板 =====
def evaluate_accuracy(model, tokenizer, dataset, max_samples=500):
    """通用准确率评估"""
    model.eval()
    correct, total = 0, 0
    
    for example in dataset[:max_samples]:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": example["question"]}],
            tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        
        completion = tokenizer.decode(output[0][inputs.input_ids.shape[1]:],
                                       skip_special_tokens=True)
        predicted = extract_last_number(completion)
        gt = float(example["answer"])
        
        if predicted is not None and math.isclose(predicted, gt, rel_tol=1e-5):
            correct += 1
        total += 1
    
    return correct / total
```

## 测验题

### Q1（模式识别，4分）
SFT 模板和 GRPO 模板有哪 3 个关键差异？

**答案**：(1) Trainer 不同（SFTTrainer vs GRPOTrainer），(2) GRPO 需要 reward_funcs，(3) 学习率不同（SFT 2e-5 vs GRPO 1e-6）。（每个 1.3 分）

### Q2（代码补全，3分）
补全以下评估代码中的 `???`：
```python
output = model.generate(**inputs, max_new_tokens=512, do_sample=???)
```
评估时 do_sample 应该是什么？为什么？

**答案**：`do_sample=False`，评估时用贪心解码获得确定性结果，保证可复现。（3分）

### Q3（设计题，3分）
如果你的任务不是数学推理而是代码生成，accuracy_reward 应该怎么改？

**答案**：不能简单比较数字，应该执行模型生成的代码并检查输出。例如用 subprocess 运行代码，比对 stdout 与期望输出。还要加超时和沙箱防止恶意代码。（3分）
