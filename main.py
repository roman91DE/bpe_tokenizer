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


def _get_pairs(tokens):
    """Return a dict mapping each pair to a set of indices where it occurs."""
    pairs = {}
    for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i+1])
        if pair not in pairs:
            pairs[pair] = set()
        pairs[pair].add(i)
    return pairs


def _bpe_fast(tokens: list[str], k: int, mode: str) -> list[str]:
    """
    Fast BPE implementation using pair location tracking.
    tokens: list of strings (converted to tuples internally for merging).
    """
    # Internally, tokens become list[tuple[str, ...]]
    tokens_tuple = [(t,) for t in tokens]
    total_steps = k
    step = 0
    while step < total_steps:
        pairs = _get_pairs(tokens_tuple)
        if not pairs:
            break
        # Find the most frequent pair (skip across word boundaries in 'w' mode)
        if mode == 'w':
            filtered_pairs = {pair: idxs for pair, idxs in pairs.items()
                              if not (pair[0][0].startswith('▁') and pair[1][0].startswith('▁')) and not pair[1][0].startswith('▁')}
            if not filtered_pairs:
                break
            best_pair = max(filtered_pairs.items(), key=lambda x: len(x[1]))[0]
            best_count = len(filtered_pairs[best_pair])
        else:
            best_pair = max(pairs.items(), key=lambda x: len(x[1]))[0]
            best_count = len(pairs[best_pair])
        if best_count < 2:
            break
        # Merge all occurrences of best_pair
        i = 0
        new_tokens = []
        while i < len(tokens_tuple):
            if i < len(tokens_tuple) - 1 and (tokens_tuple[i], tokens_tuple[i+1]) == best_pair:
                new_tokens.append(tokens_tuple[i] + tokens_tuple[i+1])
                i += 2
            else:
                new_tokens.append(tokens_tuple[i])
                i += 1
        tokens_tuple = new_tokens
        step += 1
    # Flatten tokens back to strings
    return [''.join(t) for t in tokens_tuple]


def bpe(text: str, k: int, mode: str = "c", normalize: bool = False) -> tuple[list[tuple[str, int]], list[str]]:
    """
    Applies Byte Pair Encoding (BPE) to the input text for k steps.
    mode: 'c' for char-level, 'w' for word-boundary-aware BPE.
    normalize: if True, lowercases the input text before processing.
    Returns the frequency table and the final list of tokens.
    Shows a progress bar, updating every 100 steps.
    """
    if normalize:
        text = text.lower()
    if mode == "w":
        tokens = _get_word_boundary_tokens(text)
    else:
        tokens = _get_chars(text)
    total_steps = k
    tokens_tuple = [(t,) for t in tokens]
    step = 0
    with tqdm(total=total_steps//100 + (1 if total_steps % 100 else 0), desc="BPE Steps", unit="100 steps") as pbar:
        while step < total_steps:
            pairs = _get_pairs(tokens_tuple)
            if not pairs:
                break
            if mode == 'w':
                filtered_pairs = {pair: idxs for pair, idxs in pairs.items()
                                  if not (pair[0][0].startswith('▁') and pair[1][0].startswith('▁')) and not pair[1][0].startswith('▁')}
                if not filtered_pairs:
                    break
                best_pair = max(filtered_pairs.items(), key=lambda x: len(x[1]))[0]
                best_count = len(filtered_pairs[best_pair])
            else:
                best_pair = max(pairs.items(), key=lambda x: len(x[1]))[0]
                best_count = len(pairs[best_pair])
            if best_count < 2:
                break
            i = 0
            new_tokens = []
            while i < len(tokens_tuple):
                if i < len(tokens_tuple) - 1 and (tokens_tuple[i], tokens_tuple[i+1]) == best_pair:
                    new_tokens.append(tokens_tuple[i] + tokens_tuple[i+1])
                    i += 2
                else:
                    new_tokens.append(tokens_tuple[i])
                    i += 1
            tokens_tuple = new_tokens
            step += 1
            if step % 100 == 0 or step == total_steps:
                pbar.update(1)
    tokens = [''.join(t) for t in tokens_tuple]
    freq_table = _get_frequency_table(tokens)
    return (freq_table, tokens)
