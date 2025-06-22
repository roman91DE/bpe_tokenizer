# BPE Tokenizer

A simple yet efficient implementation of the Byte Pair Encoding (BPE) algorithm for text tokenization. This Python module can be used as a standalone command-line tool or imported as a library in your projects.

## What is BPE?

Byte Pair Encoding is a data compression technique that iteratively replaces the most frequent pair of bytes (or characters) in a sequence with a single, unused byte (or character). In the context of natural language processing, it's used as a subword tokenization method, creating a vocabulary that includes both whole words and subword units.

## Installation

No installation is required beyond Python dependencies:

```bash
pip install tqdm
```

## Command Line Usage

The BPE tokenizer can be run directly from the command line:

```bash
python bpe.py [input] [-n NSTEPS] [-o OUTPUT] [-r]
```

### Arguments

- `input` - Path to the input file (default: `-` for stdin)
- `-n, --nsteps` - Number of BPE steps to perform (default: 100)
- `-o, --output` - Output file prefix for vocabulary, tokens, and frequency table
- `-r, --relative` - Use relative frequencies instead of absolute counts
- `-l, --lowercase` - Convert input text to lowercase before processing

### Examples

Process a file and output tokenized text to stdout:
```bash
python bpe.py corpora/sherlock.txt
```

Process stdin and output tokenized text to stdout:
```bash
cat corpora/sherlock.txt | python bpe.py
```

Perform 200 BPE steps on a file:
```bash
python bpe.py corpora/sherlock.txt -n 200
```

Save outputs to files:
```bash
python bpe.py corpora/sherlock.txt -o output/sherlock
```
This generates three files:
- `output/sherlock.vocab.json` - The vocabulary as a JSON list
- `output/sherlock.tokens.txt` - The tokenized text
- `output/sherlock.freqs.json` - The frequency table

## Using as a Library

You can import the `bpe` function from the module:

```python
from bpe import bpe

# Read your text
with open("your_text_file.txt", "r") as f:
    text = f.read()

# Perform BPE
vocab, tokens, freq_table = bpe(
    text=text,       # The input text
    nsteps=100,      # Number of BPE steps
    relative_freqs=False  # Whether to use relative frequencies
)

# Access the results
print(f"Vocabulary size: {len(vocab)}")
print(f"First 10 tokens in first word: {tokens[0][:10]}")
print(f"Most common token: {max(freq_table, key=freq_table.get)}")
```

## How It Works

1. **Initialization**: The text is split into words, and each word is represented as a list of individual characters.

2. **Iterative Merging**: For `nsteps` iterations:
   - Count the frequency of all adjacent token pairs
   - Find the most frequent pair
   - Replace all occurrences of that pair with a new token (the concatenation of the pair)
   - Add this new token to the vocabulary

3. **Output**: 
   - A vocabulary (set of tokens)
   - Tokenized text (list of lists, where each inner list represents a word)
   - A frequency table for all tokens

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE.md](LICENSE.md) file for details.