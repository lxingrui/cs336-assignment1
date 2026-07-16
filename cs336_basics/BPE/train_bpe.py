import regex
from collections import defaultdict
import heapq
class PriorityItem:
    def __init__(self, val: int, key: tuple[bytes, bytes]):
        self.val = val
        self.key = key

    def __lt__(self, other):
        # 双重排序规则：val 大的优先；val 相同时，key 字典序大的优先
        if self.val != other.val:
            return self.val > other.val
        return self.key > other.key

class DynamicMaxTracker:
    def __init__(self):
        self.heap = []
        self.finder = {}  # 记录有效的 key 和最新的 val

    def add_or_update(self, key: tuple[bytes, bytes], val: int):
        """添加新元素或更新旧元素"""
        self.finder[key] = val
        heapq.heappush(self.heap, PriorityItem(val, key))

    def remove(self, key: tuple[bytes, bytes]):
        """手动删除旧元素"""
        if key in self.finder:
            del self.finder[key]

    def pop_max(self) -> tuple[tuple[bytes, bytes], int] | None:
        """【核心】提取 val 最大（且字典序最大）的元素，并自动将其删除"""
        while self.heap:
            item = heapq.heappop(self.heap)  # 弹出堆顶
            # 检查这个堆顶元素是否依然有效（没有被手动删除，且 val 是最新的）
            if item.key in self.finder and self.finder[item.key] == item.val:
                del self.finder[item.key]    # 提取的同时，从字典中删除它
                return item.key, item.val    # 返回结果
        return None  # 如果数据结构空了，返回 None
    
def init_bpe(vocab_size: int,special_tokens: list[str]):
    vocab={i:bytes([i]) for i in range(256)}
    start = 256
    if 256+len(special_tokens) > vocab_size:
        raise ValueError(f"vocab_size is too small")
    for i in special_tokens:
        vocab[start]=i.encode("utf-8")
        start += 1
    return vocab

# 预先全局创建 256 个单字节的查找表，避免在循环中重复调用 bytes([b])
BYTES_TABLE = [bytes([i]) for i in range(256)]

def get_freq_counts(text: str, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    word_freqs = defaultdict(int)
    GPT2_PAT = regex.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    
    if not special_tokens:
        text_drop = [text]
    else:
        sorted_special = sorted(special_tokens, key=len, reverse=True)
        pattern = "|".join(map(regex.escape, sorted_special))
        text_drop = regex.split(pattern, text)
        
    for chunk in text_drop:
        # 使用 findall 直接返回 list[str]，省去 Match 对象和 .group() 的调用
        pre_tokens = GPT2_PAT.findall(chunk)
        for word in pre_tokens:
            word_encoded = word.encode('utf-8')
            # 列表推导式 + 查表法，完全消除生成器开销和 bytes 构造开销
            word_tuple = tuple([BYTES_TABLE[b] for b in word_encoded])
            word_freqs[word_tuple] += 1
            
    return word_freqs

def get_pair_counts(word_freqs):
    # 使用 defaultdict 初始化计数器，默认值为 0
    result_counts = defaultdict(int)
    
    for tuple_key, count in word_freqs.items():
        # 利用 zip 错位相配，快速生成相邻的 pair

        for pair in zip(tuple_key, tuple_key[1:]):
            result_counts[pair] += count
    
    return result_counts



    
from collections import defaultdict

def merge_vocab_words(
    best_pair: tuple[bytes, bytes], 
    word_freqs: dict[tuple[bytes, ...], int]
) -> dict[tuple[bytes, ...], int]:
    
    new_word_freqs = defaultdict(int)
    b1, b2 = best_pair
    
    for word_tuple, count in word_freqs.items():
        new_word = []
        i = 0
        while i < len(word_tuple):
            # 检查是否遇到了需要合并的相邻 pair
            if i < len(word_tuple) - 1 and word_tuple[i] == b1 and word_tuple[i+1] == b2:
                new_word.append(b1 + b2)
                i += 2  # 跳过已被合并的第二部分
            else:
                new_word.append(word_tuple[i])
                i += 1
        
        # 将新组成的元组放进新词频表，并累加原有的频次
        new_word_freqs[tuple(new_word)] += count
        
    return new_word_freqs
            


