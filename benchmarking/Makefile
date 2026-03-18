profile-mem:
	python profile-mem.py
	echo "Converting memory snapshot to HTML"
	python _memory_viz.py trace_plot benchmarking/memory/mem-snapshot.pickle -o benchmarking/memory/mem-snapshot.html

download-html-conversion-tool:
	wget https://raw.githubusercontent.com/pytorch/pytorch/refs/heads/main/torch/cuda/_memory_viz.py


NCU=$(shell which ncu)
NSYS=$(shell which nsys)
PYTHON=$(shell which python)
# Get the directory containing the Python executable (venv's bin directory)
VENV_BIN=$(shell dirname $(PYTHON))
N_HIDDEN_STATES=1
N_RUNS_BENCHMARK=1
# Can be set to 'sudo' or to nothing
SUDO=
NSIGHT_RESULTS_DIR=profiles/nsight
NCU_RESULTS_DIR=profiles/ncu/tp$(N_PROCS)
NCU_TXT_DIR=profiles/ncu-txt/tp$(N_PROCS)
CASE=small
N_PROCS=1
PROTON_RESULTS_DIR=profiles/proton

speed-test-topp-topk:
	python speed_test.py --n_hidden_states=$(N_HIDDEN_STATES) --top_k=20 --top_p=0.9 --name=fused-topk,naive-compiled,flashinfer:top_k_top_p_sampling_from_logits

# NCU profiling targets:
# --import-source yes: Embeds source code into the .ncu-rep file, allowing it to be
#   viewed on a different machine without path remapping. The source is embedded
#   regardless of where you open the report.
# --source-folders: Comma-separated search paths for source files to import into report.
#   For Triton kernels, this helps NCU find the Python source files.
ncu-profile-fused-triton:
	mkdir -p $(NCU_RESULTS_DIR)
	$(SUDO) $(NCU) --set full \
		--nvtx \
		--nvtx-include "kernel/" \
		--kernel-name fused_mm_sample_triton_kernel \
		--source-folders $$(pwd)/../src/fused_mm_sampling \
		--import-source yes \
		-fo $(NCU_RESULTS_DIR)/fused-triton $(PYTHON) ./speed_test.py --name fused-triton --n_hidden_states $(N_HIDDEN_STATES) --n_runs_benchmark $(N_RUNS_BENCHMARK) --bench_fn=own


ncu-profile-flashinfer:
	mkdir -p $(NCU_RESULTS_DIR)
	$(SUDO) env "PATH=$$PATH:$(VENV_BIN)" $(NCU) --set full \
		--nvtx \
		--nvtx-include "kernel/" \
		-fo $(NCU_RESULTS_DIR)/flashinfer $(PYTHON) ./speed_test.py --name flashinfer:sampling_from_logits --n_hidden_states $(N_HIDDEN_STATES) --n_runs_benchmark $(N_RUNS_BENCHMARK) --bench_fn=own


ncu-profile-naive-compiled:
	mkdir -p $(NCU_RESULTS_DIR)
	$(SUDO) $(NCU) --set full \
		--nvtx \
		--nvtx-include "kernel/" \
		-fo $(NCU_RESULTS_DIR)/naive-compiled $(PYTHON) ./speed_test.py --name naive-compiled --n_hidden_states $(N_HIDDEN_STATES) --n_runs_benchmark $(N_RUNS_BENCHMARK) --bench_fn=own


nsight-profile-fused-triton:
	mkdir -p $(NSIGHT_RESULTS_DIR)
	$(SUDO) $(NSYS) profile \
		--force-overwrite true \
		-o $(NSIGHT_RESULTS_DIR)/fused-triton $(PYTHON) ./speed_test.py --name fused-triton --n_hidden_states $(N_HIDDEN_STATES) --n_runs_benchmark $(N_RUNS_BENCHMARK)

nsight-profile-naive-compiled:
	mkdir -p $(NSIGHT_RESULTS_DIR)
	$(SUDO) $(NSYS) profile \
		--force-overwrite true \
		-o $(NSIGHT_RESULTS_DIR)/naive-compiled $(PYTHON) ./speed_test.py --name naive-compiled --n_hidden_states $(N_HIDDEN_STATES) --n_runs_benchmark $(N_RUNS_BENCHMARK)

nsight-profile-flashinfer:
	mkdir -p $(NSIGHT_RESULTS_DIR)
	$(SUDO) $(NSYS) profile \
		--force-overwrite true \
		-o $(NSIGHT_RESULTS_DIR)/flashinfer $(PYTHON) ./speed_test.py --name flashinfer:sampling_from_logits --n_hidden_states $(N_HIDDEN_STATES) --n_runs_benchmark $(N_RUNS_BENCHMARK)

nsight-profile-naive-pt:
	mkdir -p $(NSIGHT_RESULTS_DIR)
	$(SUDO) $(NSYS) profile \
		--force-overwrite true \
		-o $(NSIGHT_RESULTS_DIR)/naive-pt $(PYTHON) ./speed_test.py --name naive-pt --n_hidden_states $(N_HIDDEN_STATES) --n_runs_benchmark $(N_RUNS_BENCHMARK)

NAME :=
NCU_METRICS=--set full
NCU_OUTPUT_FLAGS=
ncu-export:
	@test -n "$(NAME)" || (echo "Usage: make ncu-export NAME=fused-cutlass-evt"; exit 1)
	mkdir -p $(NCU_TXT_DIR)
	$(SUDO) $(NCU) $(NCU_METRICS) \
		$(NCU_OUTPUT_FLAGS) \
		--nvtx \
		--nvtx-include "kernel/" \
		--target-processes all \
		$(PYTHON) ./speed_test.py --name $(NAME) --n_hidden_states $(N_HIDDEN_STATES) --n_runs_benchmark $(N_RUNS_BENCHMARK) --bench_fn=own --case=$(CASE) --n_procs=$(N_PROCS) > $(NCU_TXT_DIR)/$(NAME).txt

ncu-export-fused-triton:
	$(MAKE) ncu-export NAME=fused-triton

ncu-export-naive-tl-matmul:
	$(MAKE) ncu-export NAME=naive-tl-matmul

ncu-export-naive-pt:
	$(MAKE) ncu-export NAME=naive-pt

ncu-export-naive-compiled:
	$(MAKE) ncu-export NAME=naive-compiled

ncu-export-flashinfer:
	$(MAKE) ncu-export NAME="flashinfer:sampling_from_logits"

ncu-export-flashinfer-top-k-top-p:
	$(MAKE) ncu-export NAME="flashinfer:top_k_top_p_sampling_from_logits"

ncu-profile-helion:
	mkdir -p $(NCU_RESULTS_DIR)
	$(SUDO) $(NCU) --set full \
		--nvtx \
		--nvtx-include "kernel/" \
		--kernel-name _helion_fused_sample_helion_kernel \
		--import-source yes \
		-fo $(NCU_RESULTS_DIR)/helion $(PYTHON) ./speed_test.py --name helion --n_hidden_states $(N_HIDDEN_STATES) --n_runs_benchmark $(N_RUNS_BENCHMARK) --bench_fn=own

ncu-export-helion:
	$(MAKE) ncu-export NAME=helion

ncu-export-all: ncu-export-fused-triton ncu-export-naive-compiled ncu-export-flashinfer ncu-export-naive-tl-matmul
	@echo "NCU reports exported to profiles/ncu-txt/"

# Batch-size sweeps: profile all methods across multiple batch sizes for a
# given config. Runs sequentially (GPU contention). Reuses ncu-export-* targets.
# Usage:
#   make sweep-bsz-ncu                          # default: case=small
#   make sweep-bsz-ncu CASE=large
#   make sweep-bsz-ncu SWEEP_BSZ="1 8 32"
#   make sweep-bsz-proton
SWEEP_BSZ=1 4 16 64 128 256
SWEEPS_DIR=profiles/sweeps/bsz
NCU_SWEEP_DIR=$(SWEEPS_DIR)/ncu-txt/tp$(N_PROCS)/case-$(CASE)
NCU_SWEEP_TARGETS=ncu-export-fused-triton ncu-export-naive-pt ncu-export-naive-compiled ncu-export-flashinfer ncu-export-flashinfer-top-k-top-p

sweep-bsz-ncu:
	@for bsz in $(SWEEP_BSZ); do \
		for tgt in $(NCU_SWEEP_TARGETS); do \
			echo "=== $$tgt, case=$(CASE), bsz=$$bsz ==="; \
			$(MAKE) $$tgt N_HIDDEN_STATES=$$bsz NCU_TXT_DIR=$(NCU_SWEEP_DIR)/bsz$$bsz CASE=$(CASE) N_PROCS=$(N_PROCS) NCU_METRICS="--metrics gpu__time_duration.sum" NCU_OUTPUT_FLAGS="--csv --page raw"; \
		done; \
	done
	@echo "NCU sweep done. Reports in $(NCU_SWEEP_DIR)/"
	$(MAKE) parse-sweep-ncu

parse-sweep-ncu:
	$(PYTHON) parse_ncu_sweep.py --dir $(NCU_SWEEP_DIR) | tee $(NCU_SWEEP_DIR)/summary.txt

parse-sweep-proton:
	@echo "TODO: parse proton chrome traces from $(PROTON_SWEEP_DIR)/"

plot-bsz-sweep-runtime:
	$(PYTHON) plot_bsz_sweep_runtime.py --n_procs $(N_PROCS) --case $(CASE) --fmt pdf --use-name-flashsampling

sweep-bsz-all: sweep-bsz-ncu sweep-bsz-proton plot-bsz-sweep-runtime

TRITON_BENCH_DIR=profiles/triton-bench/tp$(N_PROCS)

triton-benchmark:
	@if [ -d $(TRITON_BENCH_DIR) ]; then rm -rf $(TRITON_BENCH_DIR); fi
	mkdir -p $(TRITON_BENCH_DIR)
	PYTHONUNBUFFERED=1 $(PYTHON) triton_benchmark.py --tgt_dir $(TRITON_BENCH_DIR) --case $(CASE) --n_procs $(N_PROCS) $(if $(NAME),--name $(NAME)) 2>&1 | tee $(TRITON_BENCH_DIR)/logs.txt
	$(PYTHON) plot-triton-bench.py --tgt_dir $(TRITON_BENCH_DIR)
	$(PYTHON) plot-triton-bench.py --tgt_dir $(TRITON_BENCH_DIR) --fmt pdf

distr-triton-benchmark:
	$(MAKE) triton-benchmark N_PROCS=2 NAME=fused-triton,naive-pt,naive-compiled,flashinfer:sampling_from_logits,flashinfer:top_k_top_p_sampling_from_logits

clean-triton-autotunes:
	rm -rf ~/.triton/cache

nsight-comparison:
	make nsight-profile-fused-triton N_HIDDEN_STATES=256 NSIGHT_RESULTS_DIR=profiles/nsight-bsz256
	make nsight-profile-flashinfer N_HIDDEN_STATES=256 NSIGHT_RESULTS_DIR=profiles/nsight-bsz256
	make nsight-profile-fused-triton N_HIDDEN_STATES=1 NSIGHT_RESULTS_DIR=profiles/nsight-bsz1
	make nsight-profile-flashinfer N_HIDDEN_STATES=1 NSIGHT_RESULTS_DIR=profiles/nsight-bsz1

ncu-comparison:
	make ncu-profile-fused-triton N_HIDDEN_STATES=256 NCU_RESULTS_DIR=profiles/ncu-bsz256
	make ncu-profile-flashinfer N_HIDDEN_STATES=256 NCU_RESULTS_DIR=profiles/ncu-bsz256
	make ncu-profile-fused-triton N_HIDDEN_STATES=1 NCU_RESULTS_DIR=profiles/ncu-bsz1
	make ncu-profile-flashinfer N_HIDDEN_STATES=1 NCU_RESULTS_DIR=profiles/ncu-bsz1

# ── Proton intra-kernel profiling (TTGIR override workflow) ──
# 3-step process recommended by Triton team:
#   1. Dump TTGIR (make proton-dump-ttgir)
#   2. Inject proton.record statements (make proton-inject)
#   3. Run with TTGIR override (make proton-run)
# Or run all three steps at once: make proton-profile
PROTON_TTGIR_DIR=$(PROTON_RESULTS_DIR)/ttgir_dump
# Override vocab_size for profiling (smaller V → smaller trace file).
# Empty = use the case's default V.
PROTON_VOCAB_SIZE=
PROTON_VOCAB_SIZE_FLAG=$(if $(PROTON_VOCAB_SIZE),--vocab_size=$(PROTON_VOCAB_SIZE))

proton-profile: proton-clean proton-dump-ttgir proton-inject proton-run

proton-clean:
	rm -rf $(PROTON_RESULTS_DIR)

proton-dump-ttgir:
	mkdir -p $(PROTON_RESULTS_DIR)
	TRITON_DUMP_DIR=$(PROTON_TTGIR_DIR) ./dump_ttgir.sh $(PYTHON) proton_profile.py --n_hidden_states=$(N_HIDDEN_STATES) $(PROTON_VOCAB_SIZE_FLAG)

proton-inject:
	$(PYTHON) insert_proton_records.py --dir $(PROTON_TTGIR_DIR)

proton-run:
	mkdir -p $(PROTON_RESULTS_DIR)
	TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_OVERRIDE=1 TRITON_OVERRIDE_DIR=$(PROTON_TTGIR_DIR) \
		$(PYTHON) proton_profile.py --n_hidden_states=$(N_HIDDEN_STATES) $(PROTON_VOCAB_SIZE_FLAG) \
		2>&1 | tee $(PROTON_RESULTS_DIR)/proton-run.txt
	mv kernel.chrome_trace $(PROTON_RESULTS_DIR)/kernel.chrome_trace
	$(PYTHON) parse_proton_intrakernel.py $(PROTON_RESULTS_DIR)/kernel.chrome_trace | tee $(PROTON_RESULTS_DIR)/breakdown.txt

PROTON_SWEEP_DIR=$(SWEEPS_DIR)/proton/tp$(N_PROCS)/case-$(CASE)

sweep-bsz-proton:
	@for bsz in $(SWEEP_BSZ); do \
		echo "=== proton-profile, case=$(CASE), bsz=$$bsz ==="; \
		$(MAKE) proton-profile N_HIDDEN_STATES=$$bsz PROTON_RESULTS_DIR=$(PROTON_SWEEP_DIR)/bsz$$bsz CASE=$(CASE); \
	done
	@echo "Proton sweep done. Reports in $(PROTON_SWEEP_DIR)/"

COMPARISON_DIR=profiles/barrier-comparison
