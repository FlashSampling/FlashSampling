RESULTS_DIR := $(CURDIR)
VLLM := $(shell which vllm 2>/dev/null)
ifeq ($(VLLM),)
$(error vllm not found. Activate your venv first)
endif
PYTHON := $(shell which python)
BENCH_PARAMS := $(RESULTS_DIR)/bench-params.json

MODEL := openai/gpt-oss-120b
MODEL_SLUG := $(lastword $(subst /, ,$(MODEL)))
MAX_MODEL_LEN := 1024
MODEL_DIR := $(RESULTS_DIR)/$(MODEL_SLUG)

SERVE_FLAGS := --max-model-len $(MAX_MODEL_LEN) --no-enable-prefix-caching --uvicorn-log-level warning
BENCH_DATASET_FLAGS := --dataset-name hf --dataset-path AI-MO/aimo-validation-aime --temperature 0.6 --top-k -1 --top-p 1.0 --backend openai-chat --endpoint /v1/chat/completions
BENCH_FLAGS := $(BENCH_DATASET_FLAGS) --hf-output-len 256

AFTER_BENCH := curl -s -X POST http://localhost:8000/reset_prefix_cache && curl -s -X POST http://localhost:8000/reset_mm_cache && curl -s -X POST http://localhost:8000/reset_encoder_cache

SWEEP_FLAGS := --bench-params "$(BENCH_PARAMS)" --after-bench-cmd "$(AFTER_BENCH)" --num-runs 5 --show-stdout
QUICK_BENCH_PARAMS := $(RESULTS_DIR)/quick-bench-params.json
QUICK_SWEEP_FLAGS := --bench-params "$(QUICK_BENCH_PARAMS)" --after-bench-cmd "$(AFTER_BENCH)" --num-runs 1 --show-stdout

# Move the .tmp log into the latest timestamped results directory.
# Called with ; after the sweep so it runs even if the sweep exits non-zero.
define move_log
mv "$(1).tmp" "$$(ls -td $(2)/*/ | head -1)/sweep.log" 2>/dev/null || true
endef

define collect
	$(PYTHON) "$(RESULTS_DIR)/collect_results.py" "$(MODEL_DIR)" | tee "$(MODEL_DIR)/results.txt"
endef

LM_EVAL := $(VLLM_DIR)/venv/bin/lm_eval
EVAL_TASK := gsm8k_cot_zeroshot
EVAL_MAX_MODEL_LEN := 32768
EVAL_DIR := $(RESULTS_DIR)/eval
JUDGE_DIRS := $(EVAL_DIR)/baseline $(EVAL_DIR)/baseline-fmms-params $(EVAL_DIR)/fmms-triton
# Qwen3-1.7B generation_config.json defaults + enough tokens for thinking chains:
EVAL_GEN_KWARGS := temperature=0.6 top_p=0.95 top_k=20 max_tokens=16384
# FMMS-compatible params (no top-k/top-p filtering):
EVAL_GEN_KWARGS_FMMS := temperature=0.6 top_p=1.0 top_k=-1 max_tokens=16384

# Usage: $(call run_eval,<env_prefix>,<output_subdir>,<gen_kwargs>)
# Starts vLLM in background, runs lm_eval, saves results, kills vLLM.
# If gen_kwargs (3) is omitted, defaults to EVAL_GEN_KWARGS.
define run_eval
	mkdir -p "$(EVAL_DIR)/$(2)"
	$(1) $(VLLM) serve $(MODEL) --max-model-len $(EVAL_MAX_MODEL_LEN) --no-enable-prefix-caching --uvicorn-log-level warning --reasoning-parser qwen3 \
		> "$(EVAL_DIR)/$(2)/vllm.log" 2>&1 & \
	VLLM_PID=$$!; \
	echo "Waiting for vLLM server (PID $$VLLM_PID)..."; \
	for i in $$(seq 1 120); do \
		curl -sf http://localhost:8000/health > /dev/null 2>&1 && break; \
		sleep 1; \
	done; \
	$(LM_EVAL) \
		--model local-chat-completions \
		--model_args model=$(MODEL),base_url=http://localhost:8000/v1/chat/completions,num_concurrent=32,max_retries=3,tokenized_requests=False \
		--apply_chat_template \
		--tasks $(EVAL_TASK) \
		--gen_kwargs $(if $(3),$(3),$(EVAL_GEN_KWARGS)) \
		--log_samples \
		--output_path "$(EVAL_DIR)/$(2)" \
		2>&1 | tee "$(EVAL_DIR)/$(2)/lm_eval.log"; \
	STATUS=$$?; \
	kill $$VLLM_PID 2>/dev/null; wait $$VLLM_PID 2>/dev/null; \
	exit $$STATUS
endef

# --- nsys profiling ---
# Profiles vLLM decode steps under nsys. Uses cudaProfilerApi capture range so
# nsys only records between start/stop profile calls, skipping model load,
# warmup, and CUDA graph capture. `vllm bench serve --profile` triggers
# the start/stop calls automatically.
NSYS := $(or $(shell which nsys 2>/dev/null),$(wildcard /opt/nvidia/nsight-systems/*/bin/nsys))
NSYS_GPU ?= $(shell nvidia-smi --query-gpu=gpu_name --format=csv,noheader 2>/dev/null | head -1 | sed 's/NVIDIA //;s/ /-/g')
NSYS_MODEL := openai/gpt-oss-20b
NSYS_MODEL_SLUG = $(lastword $(subst /, ,$(NSYS_MODEL)))
NSYS_DIR = $(RESULTS_DIR)/profiles/nsight/$(NSYS_GPU)/$(NSYS_MODEL_SLUG)
NSYS_MAX_TOKENS := 10
NSYS_CONCURRENCY := 256

# Usage: $(call run_nsys,<env_prefix>,<output_name>,<bench_flags>)
# Uses `vllm bench sweep serve` to manage the server lifecycle (start, health
# check, bench, shutdown). The serve-cmd wraps vllm under nsys with
# cudaProfilerApi capture. The bench-cmd runs twice: warmup (no --profile,
# not recorded by nsys) then profiled (--profile, recorded by nsys).
define run_nsys
	@if [ -z "$(NSYS)" ]; then echo "ERROR: nsys not found. Install with: sudo apt-get install -y nsight-systems-2025.5.2"; exit 1; fi
	mkdir -p "$(NSYS_DIR)"
	$(VLLM) bench sweep serve \
	    --serve-cmd "$(1) VLLM_NVTX_SCOPES_FOR_PROFILING=1 $(NSYS) profile --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite true -o $(NSYS_DIR)/$(2) $(VLLM) serve $(NSYS_MODEL) $(SERVE_FLAGS) --gpu-memory-utilization 0.90 --profiler-config.profiler=cuda" \
	    --bench-cmd "$(VLLM) bench serve $(BENCH_DATASET_FLAGS) $(3)" \
	    --bench-params "$(RESULTS_DIR)/nsys-bench-params.json" \
	    --after-bench-cmd "$(AFTER_BENCH)" \
	    --server-ready-timeout 1200 \
	    --num-runs 1 \
	    --show-stdout \
	    -o "$(NSYS_DIR)/$(2)" \
	    2>&1 | tee "$(NSYS_DIR)/$(2)-bench.log"
	@# nsys exits after capture (--capture-range-end=stop) but vllm may linger.
	@-kill $$(lsof -ti :8000) 2>/dev/null; sleep 5
	@echo "Profile saved to $(NSYS_DIR)/$(2).nsys-rep"
endef

NSYS_SINGLE_FLAGS = --hf-output-len $(NSYS_MAX_TOKENS) --num-prompts 1 --max-concurrency 1
NSYS_CONCURRENCY_FLAGS = --hf-output-len $(NSYS_MAX_TOKENS) \
	    --num-prompts $$(($(NSYS_CONCURRENCY) * 2)) \
	    --max-concurrency $(NSYS_CONCURRENCY)

nsys-baseline:
	$(call run_nsys,env,baseline,$(NSYS_SINGLE_FLAGS))

nsys-fmms-triton:
	$(call run_nsys,env VLLM_USE_FMMS_SAMPLER=1 VLLM_FMMS_PROVIDER=fused-triton,fmms-triton,$(NSYS_SINGLE_FLAGS))

nsys-fmms-flashinfer:
	$(call run_nsys,env VLLM_USE_FMMS_SAMPLER=1 VLLM_FMMS_PROVIDER=flashinfer:top_k_top_p_sampling_from_logits,fmms-flashinfer,$(NSYS_SINGLE_FLAGS))

nsys-all:
	$(MAKE) nsys-baseline nsys-fmms-triton

nsys-compare-qwen3-8b nsys-compare-qwen3-8b-baseline nsys-compare-qwen3-8b-fmms-triton: NSYS_MODEL := Qwen/Qwen3-8B
nsys-compare-qwen3-32b nsys-compare-qwen3-32b-baseline nsys-compare-qwen3-32b-fmms-triton: NSYS_MODEL := Qwen/Qwen3-32B

nsys-compare-qwen3-8b:
	$(MAKE) NSYS_MODEL=Qwen/Qwen3-8B nsys-compare-qwen3-8b-baseline nsys-compare-qwen3-8b-fmms-triton

nsys-compare-qwen3-8b-baseline:
	$(call run_nsys,env,baseline-c$(NSYS_CONCURRENCY),$(NSYS_CONCURRENCY_FLAGS))

nsys-compare-qwen3-8b-fmms-triton:
	$(call run_nsys,env VLLM_USE_FMMS_SAMPLER=1 VLLM_FMMS_PROVIDER=fused-triton,fmms-triton-c$(NSYS_CONCURRENCY),$(NSYS_CONCURRENCY_FLAGS))

nsys-compare-qwen3-32b:
	$(MAKE) NSYS_MODEL=Qwen/Qwen3-32B nsys-compare-qwen3-32b-baseline nsys-compare-qwen3-32b-fmms-triton

nsys-compare-qwen3-32b-baseline:
	$(call run_nsys,env,baseline-c$(NSYS_CONCURRENCY),$(NSYS_CONCURRENCY_FLAGS))

nsys-compare-qwen3-32b-fmms-triton:
	$(call run_nsys,env VLLM_USE_FMMS_SAMPLER=1 VLLM_FMMS_PROVIDER=fused-triton,fmms-triton-c$(NSYS_CONCURRENCY),$(NSYS_CONCURRENCY_FLAGS))

MODAL_GPU := b200
MODAL_RESULTS_DIR := $(RESULTS_DIR)/../modal-results/vllm-bench-$(MODAL_GPU)

plot-modal:
	$(PYTHON) "$(RESULTS_DIR)/plot_tpot.py" --results-dir "$(MODAL_RESULTS_DIR)"
	$(PYTHON) "$(RESULTS_DIR)/plot_tpot.py" --results-dir "$(MODAL_RESULTS_DIR)" --fmt pdf

.PHONY: all quick baseline fmms-triton fmms-flashinfer quick-baseline quick-fmms-triton quick-fmms-flashinfer eval-baseline eval-baseline-fmms-params eval-fmms-triton judge nsys-baseline nsys-fmms-triton nsys-fmms-flashinfer nsys-all nsys-compare-qwen3-8b nsys-compare-qwen3-8b-baseline nsys-compare-qwen3-8b-fmms-triton nsys-compare-qwen3-32b nsys-compare-qwen3-32b-baseline nsys-compare-qwen3-32b-fmms-triton engine-stats plot-modal

$(MODEL_DIR):
	mkdir -p "$@"

all: | $(MODEL_DIR)
	$(MAKE) baseline fmms-triton
	$(call collect)

quick: | $(MODEL_DIR)
	$(MAKE) quick-baseline quick-fmms-triton
	$(call collect)

baseline:
	$(VLLM) bench sweep serve \
		--serve-cmd "$(VLLM) serve $(MODEL) $(SERVE_FLAGS)" \
		--bench-cmd "$(VLLM) bench serve $(BENCH_FLAGS)" \
		$(SWEEP_FLAGS) \
		-o "$(MODEL_DIR)/baseline" \
		2>&1 | tee "$(MODEL_DIR)/baseline.tmp"; \
	$(call move_log,$(MODEL_DIR)/baseline,$(MODEL_DIR)/baseline)

fmms-triton:
	$(VLLM) bench sweep serve \
		--serve-cmd "env VLLM_USE_FMMS_SAMPLER=1 VLLM_FMMS_PROVIDER=fused-triton $(VLLM) serve $(MODEL) $(SERVE_FLAGS)" \
		--bench-cmd "$(VLLM) bench serve $(BENCH_FLAGS)" \
		$(SWEEP_FLAGS) \
		-o "$(MODEL_DIR)/fmms-triton" \
		2>&1 | tee "$(MODEL_DIR)/fmms-triton.tmp"; \
	$(call move_log,$(MODEL_DIR)/fmms-triton,$(MODEL_DIR)/fmms-triton)

fmms-flashinfer:
	$(VLLM) bench sweep serve \
		--serve-cmd "env VLLM_USE_FMMS_SAMPLER=1 VLLM_FMMS_PROVIDER=flashinfer:top_k_top_p_sampling_from_logits $(VLLM) serve $(MODEL) $(SERVE_FLAGS)" \
		--bench-cmd "$(VLLM) bench serve $(BENCH_FLAGS)" \
		$(SWEEP_FLAGS) \
		-o "$(MODEL_DIR)/fmms-flashinfer" \
		2>&1 | tee "$(MODEL_DIR)/fmms-flashinfer.tmp"; \
	$(call move_log,$(MODEL_DIR)/fmms-flashinfer,$(MODEL_DIR)/fmms-flashinfer)

quick-baseline:
	$(VLLM) bench sweep serve \
		--serve-cmd "$(VLLM) serve $(MODEL) $(SERVE_FLAGS) --enforce-eager" \
		--bench-cmd "$(VLLM) bench serve $(BENCH_FLAGS)" \
		$(QUICK_SWEEP_FLAGS) \
		-o "$(MODEL_DIR)/baseline" \
		2>&1 | tee "$(MODEL_DIR)/baseline.tmp"; \
	$(call move_log,$(MODEL_DIR)/baseline,$(MODEL_DIR)/baseline)

quick-fmms-triton:
	$(VLLM) bench sweep serve \
		--serve-cmd "env VLLM_USE_FMMS_SAMPLER=1 VLLM_FMMS_PROVIDER=fused-triton $(VLLM) serve $(MODEL) $(SERVE_FLAGS) --enforce-eager" \
		--bench-cmd "$(VLLM) bench serve $(BENCH_FLAGS)" \
		$(QUICK_SWEEP_FLAGS) \
		-o "$(MODEL_DIR)/fmms-triton" \
		2>&1 | tee "$(MODEL_DIR)/fmms-triton.tmp"; \
	$(call move_log,$(MODEL_DIR)/fmms-triton,$(MODEL_DIR)/fmms-triton)

quick-fmms-flashinfer:
	$(VLLM) bench sweep serve \
		--serve-cmd "env VLLM_USE_FMMS_SAMPLER=1 VLLM_FMMS_PROVIDER=flashinfer:top_k_top_p_sampling_from_logits $(VLLM) serve $(MODEL) $(SERVE_FLAGS) --enforce-eager" \
		--bench-cmd "$(VLLM) bench serve $(BENCH_FLAGS)" \
		$(QUICK_SWEEP_FLAGS) \
		-o "$(MODEL_DIR)/fmms-flashinfer" \
		2>&1 | tee "$(MODEL_DIR)/fmms-flashinfer.tmp"; \
	$(call move_log,$(MODEL_DIR)/fmms-flashinfer,$(MODEL_DIR)/fmms-flashinfer)

eval-baseline:
	$(call run_eval,env,baseline)
	$(PYTHON) "$(RESULTS_DIR)/judge_eval.py" "$(EVAL_DIR)/baseline"

eval-baseline-fmms-params:
	$(call run_eval,env,baseline-fmms-params,$(EVAL_GEN_KWARGS_FMMS))
	$(PYTHON) "$(RESULTS_DIR)/judge_eval.py" "$(EVAL_DIR)/baseline-fmms-params"

eval-fmms-triton:
	$(call run_eval,env VLLM_USE_FMMS_SAMPLER=1 VLLM_FMMS_PROVIDER=fused-triton,fmms-triton,$(EVAL_GEN_KWARGS_FMMS))
	$(PYTHON) "$(RESULTS_DIR)/judge_eval.py" "$(EVAL_DIR)/fmms-triton"

judge:
	$(PYTHON) "$(RESULTS_DIR)/judge_eval.py" $(JUDGE_DIRS)

engine-stats:
	@for log in $(MODEL_DIR)/*/; do \
		latest=$$(ls -td "$$log"*/ 2>/dev/null | head -1); \
		if [ -f "$$latest/sweep.log" ]; then \
			echo "=== $$(basename $$log) ==="; \
			$(PYTHON) "$(RESULTS_DIR)/parse_engine_stats.py" "$$latest/sweep.log" --by-concurrency; \
			echo; \
		fi; \
	done
