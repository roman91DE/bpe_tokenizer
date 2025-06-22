#!/usr/bin/env python
# coding: utf-8


import re
import logging
import sys
import json
import argparse
from tqdm import tqdm
from collections import Counter
from functools import reduce


WORD_SPLIT_RE_PATTERN = re.compile(r"\W+")


def _split_words(s: str) -> list[str]:
    """
    Split a string into words by non-word characters.

    Parameters
    ----------
    s : str
        The input string to be split.

    Returns
    -------
    list[str]
        A list of words from the input string.
    """
    return re.split(WORD_SPLIT_RE_PATTERN, s)


def _split_chars(words: list[str]) -> list[list[str]]:
    """
    Convert a list of words into a list of character lists.

    Parameters
    ----------
    words : list[str]
        A list of words to be split into characters.

    Returns
    -------
    list[list[str]]
        A list of lists, where each inner list contains the characters of a word.
        Empty words are filtered out.
    """
    return list(filter(lambda xs: len(xs) > 0, [[c for c in w] for w in words]))


def _reduce_tokens(tokens: list[list[str]]) -> tuple[bool, str, list[list[str]]]:
    """
    Perform one step of the BPE algorithm by finding and replacing the most frequent pair.

    Parameters
    ----------
    tokens : list[list[str]]
        A list of token lists representing words.

    Returns
    -------
    tuple[bool, str, list[list[str]]]
        A tuple containing:
        - bool: True if a replacement was made, False otherwise.
        - str: The replacement token if one was created, empty string otherwise.
        - list[list[str]]: The updated list of token lists after replacement.
    """
    pair_counts = Counter()
    for word in tokens:
        for i in range(len(word) - 1):
            pair = word[i] + word[i + 1]
            pair_counts[pair] += 1

    if not pair_counts:
        return False, "", tokens

    replacement_token, freq = pair_counts.most_common(1)[0]
    if freq < 2:
        return False, "", tokens

    new_tokens = []
    for w in tokens:
        i = 0
        new_word = []
        while i < len(w):
            if i < len(w) - 1 and w[i] + w[i + 1] == replacement_token:
                new_word.append(replacement_token)
                i += 2
            else:
                new_word.append(w[i])
                i += 1
        new_tokens.append(new_word)

    return True, replacement_token, new_tokens


def _frequency_table(
    vocab: set[str], tokens: list[list[str]], relative: bool
) -> dict[str, int | float]:
    """
    Calculate the frequency of each token in the vocabulary.

    Parameters
    ----------
    vocab : set[str]
        The set of tokens to calculate frequencies for.
    tokens : list[list[str]]
        A list of token lists representing words.
    relative : bool
        If True, return relative frequencies (normalized by total number of tokens),
        otherwise return absolute counts.

    Returns
    -------
    dict[str, int|float]
        A dictionary mapping each token to its frequency (relative or absolute).
    """
    flattened = reduce(lambda x, y: x + y, tokens, [])
    n = len(flattened)

    ft = {}
    for token in vocab:
        freq = flattened.count(token) / n if relative else flattened.count(token)
        ft[token] = freq

    return ft


def bpe(
    text: str, nsteps: int, relative_freqs: bool = False
) -> tuple[set[str], list[list[str]], dict[str, int | float]]:
    """
    Perform Byte Pair Encoding (BPE) on the input text.

    Parameters
    ----------
    text : str
        The input text to encode.
    nsteps : int
        The number of BPE steps to perform.
    relative_freqs : bool, optional
        Whether to return relative frequencies (default is False).

    Returns
    -------
    tuple[set[str], list[list[str]], dict[str, int|float]]
        A tuple containing:
        - set[str]: The vocabulary set of tokens.
        - list[list[str]]: The tokenized text.
        - dict[str, int|float]: Frequency table for each token.
    """
    vocab = set()
    words = _split_words(text)
    tokens = _split_chars(words)

    for w in tokens:
        for c in w:
            vocab.add(c)

    with tqdm(total=nsteps, desc="BPE Steps") as pbar:
        for nstep in range(nsteps):
            ok, new_token, tokens = _reduce_tokens(tokens)
            if not ok:
                logging.warning(f"Encoding finished early after {nstep} Iterations")
                break
            assert new_token not in vocab, (
                f"Token {new_token} already included in Vocab"
            )
            vocab.add(new_token)
            pbar.update(1)

    ft = _frequency_table(vocab, tokens, relative_freqs)

    return vocab, tokens, ft


def tokens_to_string(tokens: list[list[str]]) -> str:
    """
    Convert tokenized text back to a string representation.

    Parameters
    ----------
    tokens : list[list[str]]
        The tokenized text as a list of token lists.

    Returns
    -------
    str
        A string representation of the tokenized text.
    """
    return "\n".join(["_".join(word) for word in tokens])


def write_outputs(
    output_prefix: str,
    vocab: set[str],
    tokens: list[list[str]],
    freq_table: dict[str, int | float],
) -> None:
    """
    Write BPE outputs to files with the given prefix.

    Parameters
    ----------
    output_prefix : str
        The prefix for output filenames.
    vocab : set[str]
        The vocabulary set of tokens.
    tokens : list[list[str]]
        The tokenized text.
    freq_table : dict[str, int | float]
        Frequency table for each token.

    Returns
    -------
    None
    """
    with open(f"{output_prefix}.vocab.json", "w") as f:
        json.dump(list(vocab), f, indent=2)

    with open(f"{output_prefix}.tokens.txt", "w") as f:
        f.write(tokens_to_string(tokens))

    with open(f"{output_prefix}.freqs.json", "w") as f:
        json.dump(freq_table, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Byte Pair Encoding (BPE) tokenizer")
    parser.add_argument(
        "input",
        nargs="?",
        type=str,
        default="-",
        help="Input file path (use '-' for stdin, which is the default)",
    )
    parser.add_argument(
        "-n",
        "--nsteps",
        type=int,
        default=100,
        help="Number of BPE steps to perform (default: 100)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output file prefix for vocabulary, tokens, and frequency table",
    )
    parser.add_argument(
        "-r",
        "--relative",
        action="store_true",
        help="Use relative frequencies instead of absolute counts",
    )
    parser.add_argument(
        "-l",
        "--lowercase",
        action="store_true",
        help="Lowercase the input text before processing",
    )

    args = parser.parse_args()

    if args.input == "-":
        text = sys.stdin.read()
    else:
        with open(args.input, "r", encoding="utf-8") as f:
            text = f.read()

    if args.lowercase:
        text = text.lower()

    vocab, tokens, freq_table = bpe(
        text=text, nsteps=args.nsteps, relative_freqs=args.relative
    )

    if args.output:
        write_outputs(args.output, vocab, tokens, freq_table)
    else:
        print(tokens_to_string(tokens))
