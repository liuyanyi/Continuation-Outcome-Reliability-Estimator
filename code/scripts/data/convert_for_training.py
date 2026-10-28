import json

from datasets import load_dataset
from tqdm import tqdm

ds = load_dataset("data/train/dapo-math-17k", split="train")

suffix = "Please reason step by step, and put your final answer within \\boxed{}."
data_list = []
data_list_suffix = []
for i, data in tqdm(enumerate(ds), total=len(ds)):
    msgs = data["prompt"]
    prompt = msgs[0]["content"]
    prompt_lines = prompt.split("\n\n")
    prompt_lines = [line.strip() for line in prompt_lines if line.strip()]
    # 移除前缀
    prompt_lines.pop(0)
    # 移除后缀
    prompt_lines.pop(-1)
    prompt = "\n\n".join(prompt_lines)
    prompt_with_suffix = prompt + "\n\n" + suffix

    answer = data["label"]
    data_list.append(
        {
            "messages": [{"role": "user", "content": prompt}],
            "answer": answer,
        }
    )
    data_list_suffix.append(
        {
            "id": f"dapo_{i}",
            "messages": [{"role": "user", "content": prompt_with_suffix}],
            "answer": answer,
        }
    )


# Save in file
data_len = len(data_list)
output_file = f"data/dapo_question_only_{data_len}.jsonl"
with open(output_file, "w") as f:
    for item in data_list:
        f.write(json.dumps(item) + "\n")

output_file_suffix = f"data/dapo_question_with_suffix_{data_len}.jsonl"
with open(output_file_suffix, "w") as f:
    for item in data_list_suffix:
        f.write(json.dumps(item) + "\n")
