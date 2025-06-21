import pathlib
from collections import Counter, defaultdict
import logging
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)


def read_file(filename: pathlib.Path) -> str:
    """
    Reads the content of a file and returns it as a string.
    """
    with open(filename, "r", encoding="utf-8") as file:
        return file.read()


def _get_chars(text: str) -> list[str]:
    """
    Converts a string into a list of its characters.
    """
    return [c for c in text]


def _get_word_boundary_tokens(text: str) -> list[str]:
    """
    Splits text into word-boundary-aware tokens for BPE.
    Each word is split into characters, with a special marker (▁) at the start of each word.
    """
    tokens = []
    for word in text.strip().split():
        tokens.append('▁')
        tokens.extend(list(word))
    return tokens


def _get_frequency_table(tokens: list[str]) -> list[tuple[str, int]]:
    """
    Builds a frequency table from a list of tokens and returns it sorted by frequency descending.
    """
    frequency_table = defaultdict(int)
    for token in tokens:
        frequency_table[token] += 1
    return sorted(frequency_table.items(), key=lambda item: item[1], reverse=True)


def _find_pair(tokens: list[str]) -> tuple[str, int]:
    """
    Finds the most frequent adjacent pair in the token list.
    Returns ("", 0) if no pairs exist.
    """
    pairs = [fst + scnd for fst, scnd in zip(tokens, tokens[1:])]
    if not pairs:
        return ("", 0)
    c = Counter(pairs)
    return c.most_common(1)[0]


def _find_pair_word(tokens: list[str]) -> tuple[str, int]:
    """
    Finds the most frequent adjacent pair in the token list, but does not merge across word boundaries (▁).
    Returns ("", 0) if no pairs exist.
    """
    pairs = []
    for i in range(len(tokens) - 1):
        # Do not merge across word boundaries
        if tokens[i].startswith('▁') and i != 0:
            continue
        if tokens[i+1].startswith('▁'):
            continue
        pairs.append(tokens[i] + tokens[i+1])
    if not pairs:
        return ("", 0)
    c = Counter(pairs)
    return c.most_common(1)[0]


def _bpe_step(tokens: list[str]) -> tuple[bool, list[str]]:
    """
    Performs one BPE merge step. Returns (False, tokens) if no merge is possible.
    """
    pair, count = _find_pair(tokens)
    if count <= 1:
        return False, tokens

    new_tokens = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] + tokens[i + 1] == pair:
            new_tokens.append(pair)
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    return True, new_tokens


def _bpe_step_word(tokens: list[str]) -> tuple[bool, list[str]]:
    """
    Performs one BPE merge step for word-boundary-aware tokens.
    """
    pair, count = _find_pair_word(tokens)
    if count <= 1:
        return False, tokens
    new_tokens = []
    i = 0
    while i < len(tokens):
        if (
            i < len(tokens) - 1
            and not tokens[i+1].startswith('▁')
            and not tokens[i].startswith('▁')
            and tokens[i] + tokens[i+1] == pair
        ):
            new_tokens.append(pair)
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    return True, new_tokens


def bpe(text: str, k: int, mode: str = "c", normalize: bool = False) -> tuple[list[tuple[str, int]], list[str]]:
    """
    Applies Byte Pair Encoding (BPE) to the input text for k steps.
    mode: 'c' for char-level, 'w' for word-boundary-aware BPE.
    normalize: if True, lowercases the input text before processing.
    Returns the frequency table and the final list of tokens.
    """
    if normalize:
        text = text.lower()
    if mode == "w":
        tokens = _get_word_boundary_tokens(text)
        bpe_step = _bpe_step_word
    else:
        tokens = _get_chars(text)
        bpe_step = _bpe_step

    for i in tqdm(range(k), desc="BPE Steps"):
        logging.info(f"BPE-Iteration {i}")
        ok, tokens = bpe_step(tokens)
        if not ok:
            logging.warning(f"Tokenizer stopped early after {i + 1} BPE-Steps")
            break

    freq_table = _get_frequency_table(tokens)

    return (freq_table, tokens)
