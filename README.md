# gowtham-gpt

A character-level GPT, implemented from scratch in PyTorch and trained end-to-end on
a plain-text corpus: embeddings, causal self-attention, transformer blocks, training
loop, and autoregressive sampling.

> Originally assembled while completing the [NeetCode ML course](https://neetcode.io) —
> see [Course exercises](#course-exercises) below for the from-scratch primitives
> (backprop, normalization, tokenization, etc.) that the model builds on conceptually.

## Quick start

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

python train.py                   # trains on data/corpus/input.txt, saves a checkpoint
python generate.py --prompt "The lighthouse"
```

Training runs on GPU automatically if available (CUDA or Apple Silicon MPS), otherwise CPU.

### Training on your own text

```bash
python train.py --data path/to/your_corpus.txt --epochs 5000 --model-dim 256 --num-blocks 6
python generate.py --prompt "Once upon a time" --new-chars 500 --temperature 0.8
```

Run `python train.py --help` / `python generate.py --help` for the full list of options
(context length, batch size, number of heads, learning rate, checkpoint path, ...).

A larger, more varied corpus and more training epochs will produce noticeably more
coherent text than the small bundled sample — the included `data/corpus/input.txt` is
a few paragraphs, meant for smoke-testing the pipeline quickly, not for producing
polished output.

### How much should you train, and on what?

- **Corpus**: any plain-text `.txt` file. A single consistent style (one author, one
  genre) generalizes better than a grab-bag of unrelated text. Public-domain books
  (Project Gutenberg) are a good source if you want to try something bigger than the
  bundled sample. A few hundred KB to a few MB is a reasonable range for this small
  architecture on a laptop.
- **Watch the loss, not just the epoch count**: `python train.py` prints loss every 100
  epochs. On a *small* corpus (like the bundled one), loss dropping near 0 means the
  model has memorized the text verbatim rather than learned generalizable patterns —
  that's expected and fine for smoke-testing, but generated text will just be
  paraphrased chunks of the input. On a larger corpus, expect loss to plateau higher
  and generated text to look more like *original* prose in that style.
- **Scale model size with corpus size**: the defaults (`--model-dim 128 --num-blocks 4
  --num-heads 4`) fit a few KB of text in about a minute on Apple Silicon/CUDA. For a
  multi-MB corpus, increase `--model-dim` and `--num-blocks` and expect training to
  take proportionally longer — there's no early stopping or validation split built in,
  so treat the epoch count as a knob to tune by eye via the printed loss curve.
- **Where the output goes**: `train.py` doesn't print generated text — it prints loss
  and saves a checkpoint to `checkpoints/model.pt` (path, weights, vocab, and config all
  bundled together). `generate.py` loads that checkpoint and prints sampled text to the
  terminal. For a persistent, chat-style view of generations, use the web dashboard below.

## Web dashboard

A local chat-style UI for talking to a trained checkpoint (train first — the dashboard
needs `checkpoints/model.pt` to exist):

```bash
python webapp/app.py
# then open http://127.0.0.1:5001
```

It's a small Flask server (`webapp/app.py`) plus a static HTML/CSS/JS frontend
(`webapp/static/`) — no build step. Each message you send is appended to a running
plain-text transcript (`You: ...\nGPT: ...`) that's fed back to the model as its
generation prompt, so it can "remember" earlier turns within the model's context-length
window. Temperature and reply length are adjustable from the settings (⚙) panel.

Keep in mind this is a text-continuation model, not an instruction-tuned chat
assistant — treat replies as free-associated continuations of the conversation so far,
not answers to questions. Quality scales with how much (and what) you trained on.

## How it works

```
text corpus
   │  CharVocabulary (data/vocab.py)
   ▼
token IDs ──► get_batch (data/loader.py) ──► random (X, Y) training windows
   │
   ▼
GPT (model/gpt.py)
   ├─ word + position embeddings
   ├─ N × TransformerBlock (model/transformer.py)
   │     ├─ MultiHeadSelfAttention (model/multi_head_attention.py)
   │     │     └─ SingleHeadAttention (model/attention.py) — causal, scaled dot-product
   │     └─ FeedForward — 4x MLP expansion, ReLU, dropout
   ├─ final LayerNorm
   └─ linear projection to vocab logits
   │
   ▼
train.py    — AdamW + cross-entropy over the shifted-by-one target
generate.py — autoregressive sampling: softmax(logits / temperature) → multinomial draw
```

The transformer block uses **Pre-LN** (LayerNorm before each sub-layer, not after) —
it trains more stably than the original "Attention Is All You Need" Post-LN layout,
which is why nearly every modern GPT-style model uses it.

## Project structure

```
train.py, generate.py   CLI entry points — training loop and text generation

model/                  GPT architecture
  attention.py             SingleHeadAttention — causal scaled dot-product attention
  multi_head_attention.py  MultiHeadSelfAttention — parallel heads + output projection
  transformer.py           TransformerBlock, FeedForward — Pre-LN residual block
  gpt.py                   GPT — embeddings + stacked blocks + vocab projection
  kv_cache.py               KVCache / CachedAttention  — optional cached-generation attention
  grouped_query_attention.py GroupedQueryAttention      — optional cheaper attention variant
  normalization.py          layer_norm       — from-scratch numpy reference
  batch_normalization.py    batch_norm       — from-scratch numpy reference
  rms_normalization.py      rms_norm         — from-scratch numpy reference
  embeddings.py              embedding_lookup — from-scratch numpy reference
  positional_encoding.py     sinusoidal_positional_encoding — from-scratch numpy reference

data/                   Data pipeline
  vocab.py                 CharVocabulary — the tokenizer the pipeline actually uses
  loader.py                 get_batch — random (X, Y) window sampling
  corpus/input.txt          bundled sample training text
  tokenizer.py               learn_bpe_merges — from-scratch BPE trainer (reference)
  tokenizer_utils.py         GreedyTokenizer — greedy longest-match tokenization (reference)
  dataset.py                 build_word_batches — word-level batching (reference)
  nlp_preprocessing.py       build_sentiment_dataset — toy sentiment dataset builder (reference)

foundations/            Neural network primitives, built from scratch:
                         gradient descent, activations, loss functions, a hand-rolled
                         neuron/MLP with manual backprop, weight init strategies,
                         PyTorch basics, and training diagnostics (dead ReLUs,
                         activation/gradient stats). Not used by the GPT pipeline —
                         kept as reference implementations of the underlying math.

checkpoints/             Saved model weights + vocab (created by train.py, gitignored)

webapp/                 Local chat dashboard for a trained checkpoint
  app.py                    Flask server — /api/generate, /api/info
  static/                    index.html, style.css, script.js (no build step)
```

The `model/` and `data/` reference files (numpy normalization variants, BPE, greedy
tokenization, word-level batching) aren't wired into the training pipeline — the
production model uses `torch.nn.LayerNorm` and character-level tokenization for
simplicity and speed. They're kept as working, tested from-scratch implementations of
the same ideas, useful for understanding what those PyTorch built-ins actually do.

## Course exercises

This project began as a series of exercises from the
[NeetCode ML Course](https://neetcode.io/practice?tab=coreSkills&topic=Machine+Learning):

- Math foundations — gradient descent, activations, loss functions
- Neural networks from scratch — neuron, manual backprop, MLP
- PyTorch fundamentals
- NLP pipeline — embeddings, tokenization, attention
- Transformer architecture
- GPT model + text generation

The exercise solutions live in `foundations/` and the reference implementations noted
above; the rest of the repo has since been reworked into an actual runnable project —
duplicated code across the original exercises was consolidated into a single set of
shared model components, and a real training/generation CLI was built on top.
