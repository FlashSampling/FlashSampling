pytest-distributed:
	FMMS_TEST_DISTRIBUTED=1 pytest -s tests/test_core.py::test_sampling_distribution_tp2

update-deps:
	uv lock --upgrade  # Re-resolve all deps to latest compatible versions
	uv sync --all-extras  # Install exact versions from lockfile, including optional groups

GPU := b200
POSTFIX :=
N_PROCS := 1
CASE := all
NAME :=
# Skip "Multinomial Sampling (Eager)" from plots (it's always slower than Compiled).
# To include it: make plot-all SKIP_EAGER=
SKIP_EAGER := 1
BENCH_DIR := triton-bench-$(GPU)$(POSTFIX)/tp$(N_PROCS)
RESULTS_DIR := benchmarking/modal-results/$(BENCH_DIR)
PLOT_EXTRA_FLAGS := $(if $(SKIP_EAGER),--skip_multinomial_eager=1,)

modal-speed-test:
	modal run -m src.fused_mm_sampling.modal_lib.modal_speed_test

modal-triton-benchmark: modal-create-results-triton-bench modal-get-results-triton-bench modal-plot-triton-bench

modal-create-results-triton-bench:
	mkdir -p $(RESULTS_DIR)
	GPU=$(GPU) TGT_DIR="/vol-fused-mm-sample/$(BENCH_DIR)" \
	N_PROCS=$(N_PROCS) CASE=$(CASE) NAME=$(NAME) \
	modal run \
		-m src.fused_mm_sampling.modal_lib.modal_triton_benchmark \
		> $(RESULTS_DIR)/logs.txt

modal-plot-triton-bench:
	python benchmarking/plot-triton-bench.py --tgt_dir $(RESULTS_DIR) $(PLOT_EXTRA_FLAGS)
	python benchmarking/plot-triton-bench.py --tgt_dir $(RESULTS_DIR) --fmt pdf --use_name_flashsampling=1 $(PLOT_EXTRA_FLAGS)

TRITON_BENCH_GPUS := b300 b200 h200 h100!

modal-triton-benchmark-all-gpus: modal-create-results-triton-bench-all-gpus modal-get-and-plot-triton-bench-all-gpus

modal-create-results-triton-bench-all-gpus:
	$(foreach gpu,$(TRITON_BENCH_GPUS),\
		$(MAKE) modal-create-results-triton-bench GPU=$(gpu) &) wait

modal-get-and-plot-triton-bench-all-gpus:
	$(foreach gpu,$(TRITON_BENCH_GPUS),\
		$(MAKE) modal-get-results-triton-bench modal-plot-triton-bench GPU=$(gpu) &&) true

modal-distr-triton-benchmark:
	$(MAKE) modal-triton-benchmark N_PROCS=2 NAME=fused-triton,naive-pt,naive-compiled,flashinfer:sampling_from_logits,flashinfer:top_k_top_p_sampling_from_logits

DIAGRAM_SRC := imgs/baseline-vs-fmms-diagram.drawio
DIAGRAM_PNG := imgs/baseline-vs-fmms-diagram.png
DIAGRAM_FLASHSAMPLING_PDF := imgs/baseline-vs-flashsampling-diagram.pdf

diagram:
	xvfb-run drawio --export --format png --scale 2 --border 10 \
		--output $(DIAGRAM_PNG) $(DIAGRAM_SRC)
	sed 's/FMMS/FlashSampling/g' $(DIAGRAM_SRC) > $(DIAGRAM_SRC).tmp
	xvfb-run -a drawio --export --format pdf --border 10 \
		--output $(DIAGRAM_FLASHSAMPLING_PDF) $(DIAGRAM_SRC).tmp
	rm $(DIAGRAM_SRC).tmp

TRITON_BENCH_TPS := 1 2

plot-all:
	$(foreach gpu,$(TRITON_BENCH_GPUS),\
		$(foreach tp,$(TRITON_BENCH_TPS),\
			$(if $(wildcard benchmarking/modal-results/triton-bench-$(gpu)/tp$(tp)/*.csv),\
				python benchmarking/plot-triton-bench.py --tgt_dir benchmarking/modal-results/triton-bench-$(gpu)/tp$(tp) $(PLOT_EXTRA_FLAGS) && \
				python benchmarking/plot-triton-bench.py --tgt_dir benchmarking/modal-results/triton-bench-$(gpu)/tp$(tp) --fmt pdf --use_name_flashsampling=1 $(PLOT_EXTRA_FLAGS) &&,)) ) true
	$(MAKE) plot-vllm-bench

plot-vllm-bench:
	python benchmarking/vllm/plot_tpot.py --results-dir $(VLLM_BENCH_DIR)
	python benchmarking/vllm/plot_tpot.py --results-dir $(VLLM_BENCH_DIR) --fmt pdf --use-name-flashsampling=1

modal-example:
	modal run -m src.fused_mm_sampling.modal_lib.modal_example

modal-get-results-speed-test:
	mkdir -p benchmarking/modal-results/
	cd benchmarking/modal-results/ && modal volume get fused-mm-sample speed-test

modal-get-results-triton-bench:
	mkdir -p $(RESULTS_DIR)
	modal volume get fused-mm-sample $(BENCH_DIR) $(dir $(RESULTS_DIR))

modal-persistent-matmul:
	GPU=$(GPU) \
	modal run -m src.fused_mm_sampling.modal_lib.modal_persistent_matmul

modal-matmul-comparison:
	GPU=$(GPU) \
	modal run -m src.fused_mm_sampling.modal_lib.modal_matmul_comparison

# --- vLLM benchmarks on Modal ---
VLLM_MODEL := openai/gpt-oss-120b
VLLM_SWEEP := quick
VLLM_VOLUME_DIR_NAME := vllm-bench-$(GPU)$(POSTFIX)
VLLM_BENCH_DIR := benchmarking/modal-results/$(VLLM_VOLUME_DIR_NAME)
VLLM_MODEL_SLUG = $(lastword $(subst /, ,$(VLLM_MODEL)))
VLLM_VARIANTS :=

modal-vllm-benchmark-full-gpt-oss-120b:
	$(MAKE) modal-vllm-benchmark VLLM_SWEEP=all VLLM_MODEL=openai/gpt-oss-120b

modal-vllm-benchmark-full-qwen3-1.7b:
	$(MAKE) modal-vllm-benchmark VLLM_SWEEP=all VLLM_MODEL=Qwen/Qwen3-1.7B

modal-vllm-benchmark-full-qwen3-8b:
	$(MAKE) modal-vllm-benchmark VLLM_SWEEP=all VLLM_MODEL=Qwen/Qwen3-8B

modal-vllm-benchmark-full-qwen3-32b:
	$(MAKE) modal-vllm-benchmark VLLM_SWEEP=all VLLM_MODEL=Qwen/Qwen3-32B

modal-vllm-benchmark: modal-create-results-vllm-bench modal-get-results-vllm-bench modal-collect-results-vllm-bench

modal-create-results-vllm-bench:
	mkdir -p $(VLLM_BENCH_DIR)/$(VLLM_MODEL_SLUG)/logs
	GPU=$(GPU) MODEL=$(VLLM_MODEL) SWEEP=$(VLLM_SWEEP) VARIANTS=$(VLLM_VARIANTS) \
	TGT_DIR="/vol-fused-mm-sample/$(VLLM_VOLUME_DIR_NAME)" \
	modal run \
		-m src.fused_mm_sampling.modal_lib.modal_vllm_benchmark \
		2>&1 | tee $(VLLM_BENCH_DIR)/$(VLLM_MODEL_SLUG)/logs/$$(date +%Y%m%d_%H%M%S).txt

modal-get-results-vllm-bench:
	mkdir -p $(VLLM_BENCH_DIR)
	set -e; tmpdir=$$(mktemp -d); \
	cd "$$tmpdir"; \
	modal volume get fused-mm-sample $(VLLM_VOLUME_DIR_NAME); \
	cp -a $(VLLM_VOLUME_DIR_NAME)/. "$(CURDIR)/$(VLLM_BENCH_DIR)/"; \
	rm -rf "$$tmpdir"

modal-collect-results-vllm-bench:
	@model_dir=$$(ls -d $(VLLM_BENCH_DIR)/$(VLLM_MODEL_SLUG)-trial* 2>/dev/null | sort -V | tail -1); \
	if [ -z "$$model_dir" ]; then model_dir=$(VLLM_BENCH_DIR)/$(VLLM_MODEL_SLUG); fi; \
	echo "Collecting results from $$model_dir"; \
	python benchmarking/vllm/collect_results.py "$$model_dir" | tee "$$model_dir/results.txt"