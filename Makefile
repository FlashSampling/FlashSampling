BREV_INSTANCE_TYPE := dmz.h100x2.pcie
BREV_IMAGE := nvidia/cuda:13.0.0-devel-ubuntu22.04

brev-create:  # Then run scripts/brev-bootstrap.sh on the instance
	brev create h100x2 --type $(BREV_INSTANCE_TYPE) --mode container \
		--container-image $(BREV_IMAGE) --jupyter=false --detached

pytest-distributed:
	FMMS_TEST_DISTRIBUTED=1 pytest -s tests/test_core.py::test_sampling_distribution_tp2

modal-verify-correctness-tp1:
	$(MAKE) modal-pytest-distributed N_PROCS=1

modal-verify-correctness-large-vocab:
	mkdir -p benchmarking/modal-results/pytest/$(GPU)/tp1
	GPU=$(GPU) modal run \
		-m src.fused_mm_sampling.modal_lib.modal_verify_correctness_large_vocab \
		--vocab-size $(VOCAB_SIZE) \
		--num-samples $(NUM_SAMPLES) \
		--samples-per-call $(SAMPLES_PER_CALL) 2>&1 | \
		tee benchmarking/modal-results/pytest/$(GPU)/tp1/large-vocab-v$(VOCAB_SIZE)-n$(NUM_SAMPLES).txt

modal-pytest-distributed:
	mkdir -p benchmarking/modal-results/pytest/$(GPU)/tp$(N_PROCS)
	GPU=$(GPU) N_PROCS=$(N_PROCS) \
		modal run -m src.fused_mm_sampling.modal_lib.modal_pytest_distributed 2>&1 | \
		tee benchmarking/modal-results/pytest/$(GPU)/tp$(N_PROCS)/sampling-distribution.txt

modal-versions:
	modal run -m src.fused_mm_sampling.modal_lib.modal_versions

CUTLASS_GATES := toolchain accumulator-layout thread-local-max warp-max cta-max cta-multi-column-max cta-boundary-max
CUTLASS_MODULE_toolchain := toolchain
CUTLASS_MODULE_accumulator-layout := accumulator_layout
CUTLASS_MODULE_thread-local-max := thread_local_max
CUTLASS_MODULE_warp-max := warp_max
CUTLASS_MODULE_cta-max := cta_max
CUTLASS_MODULE_cta-multi-column-max := cta_multi_column_max
CUTLASS_MODULE_cta-boundary-max := cta_boundary_max
CUTLASS_RESULT_toolchain := 00-toolchain
CUTLASS_RESULT_accumulator-layout := 01-accumulator-layout
CUTLASS_RESULT_thread-local-max := 02-thread-local-max
CUTLASS_RESULT_warp-max := 03-warp-max
CUTLASS_RESULT_cta-max := 04-cta-max
CUTLASS_RESULT_cta-multi-column-max := 05-cta-multi-column-max
CUTLASS_RESULT_cta-boundary-max := 06-cta-boundary-max
CUTLASS_LOG_toolchain := smoke.txt

modal-cutlass:
	@test -n "$(GATE)" || \
		(echo "Usage: make modal-cutlass GATE=<gate>"; \
		 echo "Available gates: $(CUTLASS_GATES)"; exit 1)
	@test -n "$(CUTLASS_MODULE_$(GATE))" || \
		(echo "Unknown CUTLASS gate '$(GATE)'. Choose one of: $(CUTLASS_GATES)"; exit 1)
	mkdir -p benchmarking/modal-results/cutlass/$(CUTLASS_RESULT_$(GATE))
	modal run -m src.fused_mm_sampling.modal_lib.cutlass.$(CUTLASS_MODULE_$(GATE)) 2>&1 | \
		tee benchmarking/modal-results/cutlass/$(CUTLASS_RESULT_$(GATE))/$(or $(CUTLASS_LOG_$(GATE)),log.txt)

update-deps:
	uv lock --upgrade  # Re-resolve all deps to latest compatible versions
	uv sync --all-extras  # Install exact versions from lockfile, including optional groups

GPU := b200
POSTFIX :=
N_PROCS := 1
N_HIDDEN_STATES := 1
CASE := all
NAME := default
DISABLE_COMPILE := 0
BENCH_FN := fi-cupti
VOCAB_SIZE := 32768
NUM_SAMPLES := 1000000
SAMPLES_PER_CALL := 10000
# Skip "Multinomial Sampling (Eager)" from plots (it's always slower than Compiled).
# To include it: make plot-all SKIP_EAGER=
SKIP_EAGER := 1
BENCH_DIR := triton-bench/$(BENCH_FN)/$(GPU)$(POSTFIX)/tp$(N_PROCS)
RESULTS_DIR := benchmarking/modal-results/$(BENCH_DIR)
PLOT_EXTRA_FLAGS := $(if $(SKIP_EAGER),--skip_multinomial_eager=1,)
modal-speed-test:
	mkdir -p benchmarking/modal-results/speed-test/$(GPU)/tp$(N_PROCS)
	GPU=$(GPU) N_PROCS=$(N_PROCS) NAME=$(NAME) N_HIDDEN_STATES=$(N_HIDDEN_STATES) BENCH_FN=$(BENCH_FN) \
		modal run -m src.fused_mm_sampling.modal_lib.modal_speed_test 2>&1 | tee benchmarking/modal-results/speed-test/$(GPU)/tp$(N_PROCS)/bsz$(N_HIDDEN_STATES).txt

modal-triton-benchmark: modal-create-results-triton-bench modal-upload-logs-triton-bench modal-get-results-triton-bench modal-plot-triton-bench

modal-ncu-test:
	modal run -m src.fused_mm_sampling.modal_lib.modal_ncu_test

modal-nsys-test:
	modal run -m src.fused_mm_sampling.modal_lib.modal_nsys_test
	mkdir -p benchmarking/modal-results
	modal volume get fused-mm-sample nsys-test benchmarking/modal-results

NSYS_VOL_DIR := nsys-profiles/$(GPU)/tp$(N_PROCS)/case-small/bsz$(N_HIDDEN_STATES)$(POSTFIX)
NSYS_DIR := benchmarking/modal-results/$(NSYS_VOL_DIR)
modal-nsys-profile:
	mkdir -p $(NSYS_DIR)
	GPU=$(GPU) NAME=$(NAME) N_HIDDEN_STATES=$(N_HIDDEN_STATES) CASE=small N_PROCS=$(N_PROCS) \
	POSTFIX=$(POSTFIX) N_RUNS_BENCHMARK=10 \
	modal run -m src.fused_mm_sampling.modal_lib.modal_nsys \
		> $(NSYS_DIR)/$(NAME).txt 2>&1
	modal volume get --force fused-mm-sample $(NSYS_VOL_DIR) benchmarking/modal-results/nsys-profiles/$(GPU)/tp$(N_PROCS)/case-small

NCU_REP_VOL_DIR := ncu-rep/$(GPU)/tp$(N_PROCS)/case-small/bsz$(N_HIDDEN_STATES)
NCU_REP_DIR := benchmarking/modal-results/$(NCU_REP_VOL_DIR)
modal-ncu-profile:
	mkdir -p $(NCU_REP_DIR)
	GPU=$(GPU) NAME=$(NAME) N_HIDDEN_STATES=$(N_HIDDEN_STATES) CASE=small N_PROCS=$(N_PROCS) NCU_MODE=profile \
	modal run -m src.fused_mm_sampling.modal_lib.modal_ncu \
		> $(NCU_REP_DIR)/$(NAME).txt 2>&1
	modal volume get fused-mm-sample $(NCU_REP_VOL_DIR)/$(NAME).ncu-rep $(NCU_REP_DIR)/

NCU_TXT_DIR := benchmarking/modal-results/ncu-txt/$(GPU)/tp$(N_PROCS)/case-small/bsz$(N_HIDDEN_STATES)
modal-ncu-export:
	mkdir -p $(NCU_TXT_DIR)
	GPU=$(GPU) NAME=$(NAME) N_HIDDEN_STATES=$(N_HIDDEN_STATES) CASE=small N_PROCS=$(N_PROCS) NCU_MODE=export \
	modal run -m src.fused_mm_sampling.modal_lib.modal_ncu \
		> $(NCU_TXT_DIR)/$(NAME).txt 2>&1

MEMORY_TRAFFIC_VOL_DIR := memory-traffic/$(GPU)/case-$(CASE)/bsz$(N_HIDDEN_STATES)
MEMORY_TRAFFIC_DIR := benchmarking/modal-results/$(MEMORY_TRAFFIC_VOL_DIR)
OUTPUT_NAME = $(NAME)
MEMORY_TRAFFIC_PROVIDER_VOL_DIR = $(MEMORY_TRAFFIC_VOL_DIR)/$(OUTPUT_NAME)
MEMORY_TRAFFIC_PROVIDER_DIR = $(MEMORY_TRAFFIC_DIR)/$(OUTPUT_NAME)

modal-memory-traffic:
	mkdir -p "$(MEMORY_TRAFFIC_PROVIDER_DIR)"
	GPU=$(GPU) modal run \
		-m src.fused_mm_sampling.modal_lib.modal_memory_traffic \
		--name "$(NAME)" \
		--output-name "$(OUTPUT_NAME)" \
		--case "$(CASE)" \
		--n-hidden-states $(N_HIDDEN_STATES) 2>&1 | \
		tee "$(MEMORY_TRAFFIC_PROVIDER_DIR)/log.txt"
	modal volume get --force fused-mm-sample "$(MEMORY_TRAFFIC_PROVIDER_VOL_DIR)/report.ncu-rep" "$(MEMORY_TRAFFIC_PROVIDER_DIR)/"
	modal volume get --force fused-mm-sample "$(MEMORY_TRAFFIC_PROVIDER_VOL_DIR)/traffic.csv" "$(MEMORY_TRAFFIC_PROVIDER_DIR)/"
	modal volume get --force fused-mm-sample "$(MEMORY_TRAFFIC_PROVIDER_VOL_DIR)/memory.json" "$(MEMORY_TRAFFIC_PROVIDER_DIR)/"

modal-memory-traffic-all:
	@set -e; \
	$(MAKE) modal-memory-traffic NAME=fused-triton OUTPUT_NAME=fused-triton & \
	$(MAKE) modal-memory-traffic NAME=fused-triton-ret-logits OUTPUT_NAME=fused-triton-ret-logits & \
	$(MAKE) modal-memory-traffic NAME=naive-compiled OUTPUT_NAME=multinomial-compiled & \
	$(MAKE) modal-memory-traffic NAME=flashinfer:top_k_top_p_sampling_from_logits OUTPUT_NAME=fi1 & \
	$(MAKE) modal-memory-traffic NAME=flashinfer:sampling_from_logits OUTPUT_NAME=fi2 & \
	wait
	$(MAKE) parse-memory-traffic

parse-memory-traffic:
	python benchmarking/parse_memory_traffic.py "$(MEMORY_TRAFFIC_DIR)"
modal-create-results-triton-bench:
	mkdir -p $(RESULTS_DIR)
	GPU=$(GPU) TGT_DIR="/vol-fused-mm-sample/$(BENCH_DIR)" \
	N_PROCS=$(N_PROCS) CASE=$(CASE) NAME=$(NAME) DISABLE_COMPILE=$(DISABLE_COMPILE) BENCH_FN=$(BENCH_FN) \
	modal run \
		-m src.fused_mm_sampling.modal_lib.modal_triton_benchmark \
		> $(RESULTS_DIR)/logs.txt 2>&1

modal-upload-logs-triton-bench:
	modal volume put --force fused-mm-sample $(RESULTS_DIR)/logs.txt $(BENCH_DIR)/logs.txt

modal-plot-triton-bench:
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

DIAGRAM_V2_SRC := imgs/baseline-vs-fmms-diagram-v2.drawio
DIAGRAM_V2_PNG := imgs/baseline-vs-fmms-diagram-v2.png
DIAGRAM_V2_FLASHSAMPLING_PDF := imgs/baseline-vs-flashsampling-diagram-v2.pdf

diagram:
	xvfb-run drawio --export --format png --scale 2 --border 10 \
		--output $(DIAGRAM_PNG) $(DIAGRAM_SRC)
	sed 's/FMMS/FlashSampling/g' $(DIAGRAM_SRC) > $(DIAGRAM_SRC).tmp
	xvfb-run -a drawio --export --format pdf --border 10 \
		--output $(DIAGRAM_FLASHSAMPLING_PDF) $(DIAGRAM_SRC).tmp
	rm $(DIAGRAM_SRC).tmp

diagram-v2:
	xvfb-run drawio --export --format png --scale 2 --border 10 \
		--output $(DIAGRAM_V2_PNG) $(DIAGRAM_V2_SRC)
	sed 's/FMMS/FlashSampling/g' $(DIAGRAM_V2_SRC) > $(DIAGRAM_V2_SRC).tmp
	xvfb-run -a drawio --export --format pdf --border 10 \
		--output $(DIAGRAM_V2_FLASHSAMPLING_PDF) $(DIAGRAM_V2_SRC).tmp
	rm $(DIAGRAM_V2_SRC).tmp

TRITON_BENCH_TPS := 1 2

plot-all:
	$(foreach gpu,$(TRITON_BENCH_GPUS),\
		$(foreach tp,$(TRITON_BENCH_TPS),\
			$(if $(wildcard benchmarking/modal-results/triton-bench/$(BENCH_FN)/$(gpu)/tp$(tp)/*.csv),\
				python benchmarking/plot-triton-bench.py --tgt_dir benchmarking/modal-results/triton-bench/$(BENCH_FN)/$(gpu)/tp$(tp) $(PLOT_EXTRA_FLAGS) && \
				python benchmarking/plot-triton-bench.py --tgt_dir benchmarking/modal-results/triton-bench/$(BENCH_FN)/$(gpu)/tp$(tp) --fmt pdf --use_name_flashsampling=1 $(PLOT_EXTRA_FLAGS) &&,)) ) true
	$(MAKE) plot-vllm-bench

plot-vllm-bench:
	python benchmarking/vllm/plot_tpot.py --results-dir $(VLLM_BENCH_DIR) --fmt pdf --use-name-flashsampling=1

plot-vllm-bench-tp2:
	$(MAKE) plot-vllm-bench N_PROCS=2

plot-tp-scaling:
	python benchmarking/plot_tp_scaling.py --gpu $(GPU) --bench_fn own --case large --use_reruns=true

modal-example:
	modal run -m src.fused_mm_sampling.modal_lib.modal_example

modal-get-results-speed-test:
	mkdir -p benchmarking/modal-results/
	cd benchmarking/modal-results/ && modal volume get fused-mm-sample speed-test

modal-get-results-triton-bench:
	mkdir -p $(RESULTS_DIR)
	modal volume get --force fused-mm-sample $(BENCH_DIR) $(dir $(RESULTS_DIR))

modal-persistent-matmul:
	GPU=$(GPU) \
	modal run -m src.fused_mm_sampling.modal_lib.modal_persistent_matmul

modal-matmul-comparison:
	GPU=$(GPU) \
	modal run -m src.fused_mm_sampling.modal_lib.modal_matmul_comparison

# --- vLLM benchmarks on Modal ---
VLLM_MODEL := openai/gpt-oss-120b
VLLM_SWEEP := quick
VLLM_VOLUME_DIR_NAME := vllm-bench-$(GPU)-tp$(N_PROCS)$(POSTFIX)
VLLM_BENCH_DIR := benchmarking/modal-results/$(VLLM_VOLUME_DIR_NAME)
VLLM_MODEL_SLUG = $(lastword $(subst /, ,$(VLLM_MODEL)))
VLLM_VARIANTS :=
VLLM_RESUME_EXPERIMENT :=
MODAL_RUN_FLAGS :=

modal-vllm-benchmark-full-gpt-oss-120b:
	$(MAKE) modal-vllm-benchmark VLLM_SWEEP=all VLLM_MODEL=openai/gpt-oss-120b

modal-vllm-benchmark-full-qwen3-1.7b:
	$(MAKE) modal-vllm-benchmark VLLM_SWEEP=all VLLM_MODEL=Qwen/Qwen3-1.7B

modal-vllm-benchmark-quick-gemma-3-1b-it:
	$(MAKE) modal-vllm-benchmark VLLM_SWEEP=quick VLLM_MODEL=google/gemma-3-1b-it

modal-vllm-benchmark-full-gemma-3-1b-it:
	$(MAKE) modal-vllm-benchmark VLLM_SWEEP=all VLLM_MODEL=google/gemma-3-1b-it

modal-vllm-benchmark-full-qwen3-4b:
	$(MAKE) modal-vllm-benchmark VLLM_SWEEP=all VLLM_MODEL=Qwen/Qwen3-4B

modal-vllm-benchmark-quick-qwen3-4b:
	$(MAKE) modal-vllm-benchmark VLLM_SWEEP=quick VLLM_MODEL=Qwen/Qwen3-4B

modal-vllm-benchmark-full-qwen3-8b:
	$(MAKE) modal-vllm-benchmark VLLM_SWEEP=all VLLM_MODEL=Qwen/Qwen3-8B

modal-vllm-benchmark-full-qwen3-32b:
	$(MAKE) modal-vllm-benchmark VLLM_SWEEP=all VLLM_MODEL=Qwen/Qwen3-32B

modal-vllm-benchmark-full-qwen3-32b-tp2:
	$(MAKE) modal-vllm-benchmark VLLM_SWEEP=all VLLM_MODEL=Qwen/Qwen3-32B N_PROCS=2

modal-vllm-benchmark-quick-qwen3-32b-tp2:
	$(MAKE) modal-vllm-benchmark VLLM_SWEEP=quick VLLM_MODEL=Qwen/Qwen3-32B N_PROCS=2

modal-vllm-benchmark-full-llama-3.3-70b-tp2:
	$(MAKE) modal-vllm-benchmark VLLM_SWEEP=all VLLM_MODEL=meta-llama/Llama-3.3-70B-Instruct N_PROCS=2

modal-vllm-benchmark-quick-llama-3.3-70b-tp2:
	$(MAKE) modal-vllm-benchmark VLLM_SWEEP=quick VLLM_MODEL=meta-llama/Llama-3.3-70B-Instruct N_PROCS=2

modal-vllm-benchmark: modal-create-results-vllm-bench modal-get-results-vllm-bench modal-collect-results-vllm-bench

modal-create-results-vllm-bench:
	mkdir -p $(VLLM_BENCH_DIR)/$(VLLM_MODEL_SLUG)/logs
	GPU=$(GPU) N_PROCS=$(N_PROCS) MODEL=$(VLLM_MODEL) SWEEP=$(VLLM_SWEEP) VARIANTS=$(VLLM_VARIANTS) \
	RESUME_EXPERIMENT=$(VLLM_RESUME_EXPERIMENT) \
	TGT_DIR="/vol-fused-mm-sample/$(VLLM_VOLUME_DIR_NAME)" \
	modal run $(MODAL_RUN_FLAGS) \
		-m src.fused_mm_sampling.modal_lib.modal_vllm_benchmark \
		2>&1 | tee $(VLLM_BENCH_DIR)/$(VLLM_MODEL_SLUG)/logs/$$(date +%Y%m%d_%H%M%S).txt

modal-get-results-vllm-bench:
	mkdir -p $(VLLM_BENCH_DIR)
	set -e; tmpdir=$$(mktemp -d); \
	cd "$$tmpdir"; \
	modal volume get --force fused-mm-sample $(VLLM_VOLUME_DIR_NAME); \
	cp -a $(VLLM_VOLUME_DIR_NAME)/. "$(CURDIR)/$(VLLM_BENCH_DIR)/"; \
	rm -rf "$$tmpdir"

modal-collect-results-vllm-bench:
	@model_dir=$$(ls -d $(VLLM_BENCH_DIR)/$(VLLM_MODEL_SLUG)-trial* 2>/dev/null | sort -V | tail -1); \
	if [ -z "$$model_dir" ]; then model_dir=$(VLLM_BENCH_DIR)/$(VLLM_MODEL_SLUG); fi; \
	echo "Collecting results from $$model_dir"; \
	python benchmarking/vllm/collect_results.py "$$model_dir" | tee "$$model_dir/results.txt"
