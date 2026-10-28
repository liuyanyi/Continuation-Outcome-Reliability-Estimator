import orjson
from pathlib import Path

from transformers import AutoTokenizer

from cognitive_probe.utils.tokenize_utils import is_character_in_token

tokenizer_path = "Qwen/Qwen3-4B-Instruct-2507"


def build_track_tokens(tokenizer, track_characters: list[str]) -> dict[int, str]:
    """根据需要追踪的字符列表，构建对应的 token id 列表"""
    track_token_ids = set()
    for char in track_characters:
        for token_id in sorted(set(tokenizer.get_vocab().values())):
            if is_character_in_token(char, token_id, tokenizer):
                track_token_ids.add(token_id)
    id2text = {}
    for token_id in track_token_ids:
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        id2text[token_id] = token_text
    # sort, id顺序
    id2text = dict(sorted(id2text.items()))
    return id2text


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build a tokenizer-specific newline token map")
    parser.add_argument("--tokenizer-path", default=tokenizer_path)
    parser.add_argument("--output", default="track_token_id2text.json")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    track_characters = ["\n"]
    track_token_ids = build_track_tokens(tokenizer, track_characters)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(orjson.dumps(track_token_ids, option=orjson.OPT_INDENT_2))
