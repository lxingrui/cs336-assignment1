# ==================== 1. 导入与模块级辅助函数（文件最外层） ====================
import os
import json
from typing import Iterable, Iterator, Any
from collections import defaultdict
import regex
from functools import lru_cache

@lru_cache()
def bytes_to_unicode() -> dict[int, str]:
    """官方 GPT-2 字节到可见 Unicode 字符的映射"""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

# 提前在最外层初始化好逆向映射字典，避免每次调用函数都重新计算
byte_to_unicode_map = bytes_to_unicode()
unicode_to_byte_map = {v: k for k, v in byte_to_unicode_map.items()}

def token_str_to_bytes(token_str: str) -> bytes:
    """将 "Ġt" 等 Unicode 字符串还原为 b" t" """
    return bytes([unicode_to_byte_map[char] for char in token_str])


# ==================== 2. Tokenizer 类定义 ====================
class Tokenizer:
    def __init__(
        self, 
        vocab: dict[int, bytes], 
        merges: list[tuple[bytes, bytes]], 
        special_tokens: list[str] | None = None
    ):
        self.bytes_table = [bytes([i]) for i in range(256)]
        self.vocab = vocab.copy()
        
        # 保存特殊 Token 的各种属性（修正上一轮未定义属性的 Bug）
        self.special_tokens = special_tokens if special_tokens else []
        self.special_tokens_set = set(self.special_tokens)
        
        # 建立反向词表
        self.token_to_id = {v: k for k, v in vocab.items()}
        
        # 追加不存在的 special_tokens 到 vocab 和 token_to_id 中
        if self.special_tokens:
            for spe_token in self.special_tokens:
                spe_bytes = spe_token.encode("utf-8")
                if spe_bytes not in self.token_to_id:
                    next_id = len(self.token_to_id)
                    self.token_to_id[spe_bytes] = next_id
                    self.vocab[next_id] = spe_bytes
                    
        # 编译特殊 Token 的切分正则
        if self.special_tokens:
            sorted_special = sorted(self.special_tokens, key=len, reverse=True)
            self.special_pattern = regex.compile("(" + "|".join(map(regex.escape, sorted_special)) + ")")
        else:
            self.special_pattern = None

        # 建立合并优先级字典 (pair -> index)
        self.merge_ranks = {merge: i for i, merge in enumerate(merges)}

    def decode(self, ids: list[int]) -> str:
        byte_pieces = [self.vocab[idx] for idx in ids]
        joined_bytes = b"".join(byte_pieces)
        return joined_bytes.decode('utf-8', errors='replace')
    
    def _bpe_encode_word(self, b_word: bytes) -> list[bytes]:
        word = [self.bytes_table[b] for b in b_word]
        while len(word) > 1:
            pairs = list(zip(word, word[1:]))
            best_pair = None
            best_pair_idx = float('inf')
            
            for pair in pairs:
                cur_pair_idx = self.merge_ranks.get(pair, float('inf'))
                if cur_pair_idx < best_pair_idx:
                    best_pair = pair
                    best_pair_idx = cur_pair_idx
                    
            if best_pair_idx == float('inf') or best_pair is None:
                break
                
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == best_pair[0] and word[i+1] == best_pair[1]:
                    new_word.append(word[i] + word[i+1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
        return word
    
    def encode(self, text: str) -> list[int]:
        GPT2_PAT = regex.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        ids = []
        
        if not self.special_pattern:
            chunks = [text]
        else:
            chunks = self.special_pattern.split(text)
            
        for chunk in chunks:
            if not chunk:
                continue
                
            if chunk in self.special_tokens_set:
                # 完善上一轮的 pass 部分
                chunk_bytes = chunk.encode("utf-8")
                ids.append(self.token_to_id[chunk_bytes])
            else:
                pre_tokens = GPT2_PAT.findall(chunk)
                for word in pre_tokens:
                    b_word = word.encode('utf-8')
                    merged_bytes_list = self._bpe_encode_word(b_word)
                    for b_tok in merged_bytes_list:
                        ids.append(self.token_to_id[b_tok])
                        
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for line in iterable:
            yield from self.encode(line)

    @classmethod
    def from_files(
        cls, 
        vocab_filepath: str, 
        merges_filepath: str, 
        special_tokens: list[str] | None = None
    ) -> "Tokenizer":
        # 1. 载入 vocab.json
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_raw = json.load(f)
            
        # 2. 转换 vocab (Unicode 字符串 -> raw bytes)
        vocab = {}
        for token_str, idx in vocab_raw.items():
            vocab[int(idx)] = token_str_to_bytes(token_str)
            
        # 3. 载入 merges 并还原为 bytes 元组
        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) == 2:
                    tok1 = token_str_to_bytes(parts[0])
                    tok2 = token_str_to_bytes(parts[1])
                    merges.append((tok1, tok2))
                    
        # 4. 实例化
        return cls(vocab, merges, special_tokens)