from functools import lru_cache


@lru_cache
def is_character_in_token(character: str, token_id: int, tokenizer) -> bool:
    """检查字符是否包含在给定的token中"""
    token_str = tokenizer.decode([token_id])
    return character in token_str

@lru_cache
def token_id_to_str(token_id: int, tokenizer) -> str:
    """将token id转换为字符串"""
    return tokenizer.decode([token_id])