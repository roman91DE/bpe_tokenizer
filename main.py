import pathlib
from collections import defaultdict
import logging
from typing import Any
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
        tokens.append("▁")
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


def _get_pairs(tokens) -> dict[Any, Any]:
    """Return a dict mapping each pair to a set of indices where it occurs."""
    pairs = {}
    for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i + 1])
        if pair not in pairs:
            pairs[pair] = set()
        pairs[pair].add(i)
    return pairs


def bpe(
    text: str, k: int, mode: str = "c", normalize: bool = False
) -> tuple[list[tuple[str, int]], list[str]]:
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
    with tqdm(
        total=total_steps // 100 + (1 if total_steps % 100 else 0),
        desc="BPE Steps",
        unit="100 steps",
    ) as pbar:
        while step < total_steps:
            pairs = _get_pairs(tokens_tuple)
            if not pairs:
                break
            if mode == "w":
                filtered_pairs = {
                    pair: idxs
                    for pair, idxs in pairs.items()
                    if not (pair[0][0].startswith("▁") and pair[1][0].startswith("▁"))
                    and not pair[1][0].startswith("▁")
                }
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
                if (
                    i < len(tokens_tuple) - 1
                    and (tokens_tuple[i], tokens_tuple[i + 1]) == best_pair
                ):
                    new_tokens.append(tokens_tuple[i] + tokens_tuple[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens_tuple[i])
                    i += 1
            tokens_tuple = new_tokens
            step += 1
            if step % 100 == 0 or step == total_steps:
                pbar.update(1)
    tokens = ["".join(t) for t in tokens_tuple]
    freq_table = _get_frequency_table(tokens)
    return (freq_table, tokens)


def write_output(
    dir: pathlib.Path, freq_table: list[tuple[str, int]], tokens: list[str]
) -> None:
    """
    Writes the frequency table and tokens to files in the specified directory.
    """
    if dir.exists():
        logging.warning(f"Directory {dir} already exists. Overwriting files.")

    dir.mkdir(parents=True, exist_ok=True)
    freq_file = dir / "freq_table.txt"
    tokens_file = dir / "tokens.txt"

    with open(freq_file, "w", encoding="utf-8") as f:
        for token, freq in freq_table:
            f.write(f"{token}\t{freq}\n")

    with open(tokens_file, "w", encoding="utf-8") as f:
        f.write("\n".join(tokens))

    logging.info(f"Frequency table written to {freq_file}")
    logging.info(f"Tokens written to {tokens_file}")
