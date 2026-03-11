# LM Head Configurations Across LLMs

The fused matmul+sampling kernel operates on the LM head: a `[hidden_size, vocab_size]` weight matrix.
To make meaningful benchmark claims, we need configurations that represent real LLMs.

### Reading config.json files

The location of `vocab_size` and `hidden_size` in `config.json` depends on the model architecture:

- **Text-only models** (Llama, Qwen, Mistral, Phi, Gemma 3 1B): keys are at the **top level**.
  Architecture is typically `*ForCausalLM`. Look for top-level `vocab_size` and `hidden_size`.
- **Multimodal models** (Gemma 3 4B/12B/27B): keys are **nested under `text_config`**.
  Architecture is `*ForConditionalGeneration`. Look for `text_config.vocab_size` and
  `text_config.hidden_size`. The top-level `hidden_size` (if present) refers to the vision encoder.
- Some configs **omit `vocab_size`** entirely (e.g. the 12B Gemma 3 config on some mirrors).
  In that case, it defaults to the model class default (262208 for Gemma 3 multimodal models).

## Dense Models

| Model | config.json | Released | vocab_size | hidden_size | LM Head (bf16) |
|-------|-------------|----------|-----------|-------------|-----------------|
| Command R+ 104B | [link](https://huggingface.co/CohereForAI/c4ai-command-r-plus/blob/main/config.json) | Apr 2024 | 256,000 | 12,288 | 6.0 GB |
| Llama 3.1 405B | [link](https://huggingface.co/meta-llama/Llama-3.1-405B/blob/main/config.json) | Jul 2024 | 128,256 | 16,384 | 4.0 GB |
| Command R 35B | [link](https://huggingface.co/CohereForAI/c4ai-command-r-v01/blob/main/config.json) | Mar 2024 | 256,000 | 8,192 | 4.0 GB |
| Gemma 3 27B | [link](https://huggingface.co/google/gemma-3-27b-pt/blob/main/config.json) | Mar 2025 | 262,208 | 5,376 | 2.7 GB |
| Llama 3 70B | [link](https://huggingface.co/meta-llama/Llama-3.1-70B/blob/main/config.json) | Jul 2024 | 128,256 | 8,192 | 2.0 GB |
| Gemma 3 12B | [link](https://huggingface.co/google/gemma-3-12b-pt/blob/main/config.json) | Mar 2025 | 262,208 | 3,840 | 1.9 GB |
| Qwen3 14B | [link](https://huggingface.co/Qwen/Qwen3-14B/blob/main/config.json) | Apr 2025 | 151,936 | 5,120 | 1.5 GB |
| Qwen3 32B | [link](https://huggingface.co/Qwen/Qwen3-32B/blob/main/config.json) | Apr 2025 | 151,936 | 5,120 | 1.5 GB |
| Gemma 3 4B | [link](https://huggingface.co/google/gemma-3-4b-pt/blob/main/config.json) | Mar 2025 | 262,208 | 2,560 | 1.3 GB |
| Phi-4 Mini 3.8B | [link](https://huggingface.co/microsoft/Phi-4-mini-instruct/blob/main/config.json) | Feb 2025 | 200,064 | 3,072 | 1.2 GB |
| Qwen3 8B | [link](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json) | Apr 2025 | 151,936 | 4,096 | 1.2 GB |
| Llama 3 8B | [link](https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/config.json) | Jul 2024 | 128,256 | 4,096 | 1.0 GB |
| Phi-4 14B | [link](https://huggingface.co/microsoft/phi-4/blob/main/config.json) | Dec 2024 | 100,352 | 5,120 | 980 MB |
| Llama 3.2 3B | [link](https://huggingface.co/meta-llama/Llama-3.2-3B/blob/main/config.json) | Sep 2024 | 128,256 | 3,072 | 751 MB |
| Qwen3 4B | [link](https://huggingface.co/Qwen/Qwen3-4B/blob/main/config.json) | Apr 2025 | 151,936 | 2,560 | 742 MB |
| Gemma 3 1B | [link](https://huggingface.co/google/gemma-3-1b-pt/blob/main/config.json) | Mar 2025 | 262,144 | 1,152 | 576 MB |
| Llama 3.2 1B | [link](https://huggingface.co/meta-llama/Llama-3.2-1B/blob/main/config.json) | Sep 2024 | 128,256 | 2,048 | 501 MB |
| Qwen3 0.6B | [link](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json) | Apr 2025 | 151,936 | 1,024 | 297 MB |

## MoE Models

| Model | config.json | Released | vocab_size | hidden_size | Experts | Active/Token | LM Head (bf16) |
|-------|-------------|----------|-----------|-------------|---------|-------------|-----------------|
| Kimi K2.5 | [link](https://huggingface.co/moonshotai/Kimi-K2.5/blob/main/config.json) | Feb 2026 | 163,840 | 7,168 | 384 | 8 | 2.2 GB |
| Llama 4 Maverick | [link](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct/blob/main/config.json) | Apr 2025 | 202,048 | 5,120 | 128 | 1 | 1.9 GB |
| GLM-5 | [link](https://huggingface.co/zai-org/GLM-5/blob/main/config.json) | Feb 2026 | 154,880 | 6,144 | 256 | 8 | 1.8 GB |
| DeepSeek V3 | [link](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json) | Dec 2024 | 129,280 | 7,168 | 256 | 8 | 1.8 GB |
| Mistral Large 3 | [link](https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512) | Dec 2025 | 131,072 | 7,168 | 128 | 4 | 1.8 GB |
| DeepSeek V3.2 | [link](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/config.json) | Jun 2025 | 129,280 | 7,168 | 256 | 8 | 1.7 GB |
| Qwen3 235B-A22B | [link](https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/main/config.json) | Apr 2025 | 151,936 | 4,096 | 128 | 8 | 1.2 GB |
| GPT-OSS 120B | [link](https://huggingface.co/openai/GPT-OSS-120B/blob/main/config.json) | Aug 2025 | 201,088 | 2,880 | 128 | 4 | 1.1 GB |
| GPT-OSS 20B | [link](https://huggingface.co/openai/GPT-OSS-20B/blob/main/config.json) | Aug 2025 | 201,088 | 2,880 | 32 | 4 | 1.1 GB |
| DeepSeek V2 | [link](https://huggingface.co/deepseek-ai/DeepSeek-V2/blob/main/config.json) | May 2024 | 102,400 | 5,120 | 160 | 6 | 1.0 GB |
| Qwen3 30B-A3B | [link](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json) | Apr 2025 | 151,936 | 2,048 | 128 | 8 | 594 MB |
| Mixtral 8x22B | [link](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1/blob/main/config.json) | Apr 2024 | 32,000 | 6,144 | 8 | 2 | 375 MB |

## MoE and the LM Head

The LM head in MoE models is identical to dense models: a single `nn.Linear(hidden_size, vocab_size)`.
MoE only replaces the FFN/MLP blocks inside transformer layers with sparse expert routing.
Embeddings, attention, layer norms, and the LM head are all standard dense layers.

This means the fused matmul+sampling kernel works the same way regardless of whether the model
is dense or MoE. The key observation is that MoE models have smaller `hidden_size` relative to
their total parameter count (e.g., DeepSeek V3 at 671B total params has `hidden_size=7168`,
comparable to Llama 70B's 8192).

## Tensor Parallelism for the LM Head

When models are served with tensor parallelism (TP), the LM head weight is sharded along the
**vocab dimension**. Each GPU holds `[hidden_size, vocab_size / TP]` and computes partial logits.

Examples at TP=8:

| Model | Full vocab_size | Per-GPU vocab_size |
|-------|-----------------|--------------------|
| Llama 3 70B | 128,256 | 16,032 |
| Gemma 3 27B | 262,208 | 32,776 |
| DeepSeek V3 | 129,280 | 16,160 |

This is relevant for benchmarking: in production multi-GPU serving, each GPU operates on a
sharded vocab size (16K-64K) rather than the full 128K-256K.

## Benchmark Configurations

The tables above reveal two natural model families with distinct LM head shapes:

- **Qwen family**: V~152K, d=4,096 (Qwen3-8B, Qwen3-235B MoE)
- **Llama/DeepSeek family**: V~128K, d=8,192 (Llama 3 70B, DeepSeek V3)

Each benchmark config uses the (V, d) pair that matches its representative models.
The "small" vs "large" naming refers to the LM head memory footprint.

| Group | V | d | LM Head (bf16) | Representative Models |
| - | - | - | - | ---- |
| **Small** | 151,936 | 4,096 | 1.19 GB | Qwen3-8B, Qwen3-235B MoE |
| **Large** | 128,256 | 8,192 | 2.01 GB | Llama 3 70B, DeepSeek V3 |

Gemma 3's 262K vocabulary is an outlier and could be used as a stress test for memory-bandwidth sensitivity.

## Trends (as of early 2026)

The latest frontier MoE models (Kimi K2.5, GLM-5, DeepSeek V3.2, Mistral Large 3) cluster around d=6,144-7,168 with V=129K-164K, producing LM heads of 1.7-2.2 GB.
This aligns well with the existing "large" benchmark config (V=128K, d=8,192).
Llama 4 Maverick is an outlier with a large vocab (V=202K) but smaller hidden dim (d=5,120).
All frontier open-source models are now MoE, reinforcing that the decode-time LM head matmul is shaped by the active hidden dim (not total params).
