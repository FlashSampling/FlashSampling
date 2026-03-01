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
SUDO=sudo
NSIGHT_RESULTS_DIR=profiles/nsight
NCU_RESULTS_DIR=profiles/ncu
PROTON_RESULTS_DIR=profiles/proton

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
		-fo $(NCU_RESULTS_DIR)/fused-triton $(PYTHON) ./speed_test.py --name fused-triton --n_hidden_states $(N_HIDDEN_STATES) --n_runs_benchmark $(N_RUNS_BENCHMARK)


ncu-profile-flashinfer:
	mkdir -p $(NCU_RESULTS_DIR)
	$(SUDO) env "PATH=$$PATH:$(VENV_BIN)" $(NCU) --set full \
		--nvtx \
		--nvtx-include "kernel/" \
		-fo $(NCU_RESULTS_DIR)/flashinfer $(PYTHON) ./speed_test.py --name flashinfer:sampling_from_logits --n_hidden_states $(N_HIDDEN_STATES) --n_runs_benchmark $(N_RUNS_BENCHMARK)


ncu-profile-naive-compiled:
	mkdir -p $(NCU_RESULTS_DIR)
	$(SUDO) $(NCU) --set full \
		--nvtx \
		--nvtx-include "kernel/" \
		-fo $(NCU_RESULTS_DIR)/naive-compiled $(PYTHON) ./speed_test.py --name naive-compiled --n_hidden_states $(N_HIDDEN_STATES) --n_runs_benchmark $(N_RUNS_BENCHMARK)


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

ncu-export-fused-triton:
	mkdir -p profiles/ncu-txt
	sudo $(NCU) --set full \
		--nvtx \
		--nvtx-include "kernel/" \
		--kernel-name fused_mm_sample_triton_kernel \
		$(PYTHON) ./speed_test.py --name fused-triton --n_hidden_states $(N_HIDDEN_STATES) --n_runs_benchmark $(N_RUNS_BENCHMARK) > profiles/ncu-txt/fused-triton.txt

ncu-export-naive-tl-matmul:
	mkdir -p profiles/ncu-txt
	$(NCU) --set full \
		--nvtx \
		--nvtx-include "kernel/" \
		--kernel-name matmul_kernel \
		$(PYTHON) ./speed_test.py --name naive-tl-matmul --n_hidden_states $(N_HIDDEN_STATES) --n_runs_benchmark $(N_RUNS_BENCHMARK) > profiles/ncu-txt/naive-tl-matmul.txt

ncu-export-naive-compiled:
	mkdir -p profiles/ncu-txt
	$(NCU) --set full \
		--nvtx \
		--nvtx-include "kernel/" \
		$(PYTHON) ./speed_test.py --name naive-compiled --n_hidden_states $(N_HIDDEN_STATES) --n_runs_benchmark $(N_RUNS_BENCHMARK) > profiles/ncu-txt/naive-compiled.txt

ncu-export-flashinfer:
	mkdir -p profiles/ncu-txt
	$(NCU) --set full \
		--nvtx \
		--nvtx-include "kernel/" \
		$(PYTHON) ./speed_test.py --name flashinfer:sampling_from_logits --n_hidden_states $(N_HIDDEN_STATES) --n_runs_benchmark $(N_RUNS_BENCHMARK) > profiles/ncu-txt/flashinfer.txt

ncu-profile-helion:
	mkdir -p $(NCU_RESULTS_DIR)
	$(SUDO) $(NCU) --set full \
		--nvtx \
		--nvtx-include "kernel/" \
		--kernel-name _helion_fused_sample_helion_kernel \
		--import-source yes \
		-fo $(NCU_RESULTS_DIR)/helion $(PYTHON) ./speed_test.py --name helion --n_hidden_states $(N_HIDDEN_STATES) --n_runs_benchmark $(N_RUNS_BENCHMARK)

ncu-export-helion:
	mkdir -p profiles/ncu-txt
	$(SUDO) $(NCU) --set full \
		--nvtx \
		--nvtx-include "kernel/" \
		--kernel-name _helion_fused_sample_helion_kernel \
		$(PYTHON) ./speed_test.py --name helion --n_hidden_states $(N_HIDDEN_STATES) --n_runs_benchmark $(N_RUNS_BENCHMARK) > profiles/ncu-txt/helion.txt

ncu-export-all: ncu-export-fused-triton ncu-export-naive-compiled ncu-export-flashinfer ncu-export-naive-tl-matmul
	@echo "NCU reports exported to profiles/ncu-txt/"

CASE:=all
triton-benchmark:
	@if [ -d profiles/triton-bench/ ]; then rm -rf profiles/triton-bench/; fi
	mkdir -p profiles/triton-bench
	PYTHONUNBUFFERED=1 $(PYTHON) triton_benchmark.py --tgt_dir profiles/triton-bench/ --case $(CASE) 2>&1 | tee profiles/triton-bench/logs.txt
	$(PYTHON) plot-triton-bench.py
	$(PYTHON) plot-triton-bench.py --fmt pdf

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

PROTON_NAME=fused-triton
# CUPTI 13.1 libs matching the driver. Triton bundles CUPTI 2025.1.1 which segfaults
# with the CUDA 13+ driver. Point TRITON_CUPTI_LIB_PATH to the system CUPTI dir.
CUPTI_LIB_DIR=/usr/local/cuda-13.1/targets/x86_64-linux/lib

proton-profile:
	mkdir -p $(PROTON_RESULTS_DIR)/$(PROTON_NAME)
	TRITON_CUPTI_LIB_PATH=$(CUPTI_LIB_DIR) $(PYTHON) ./speed_test.py --n_hidden_states=$(N_HIDDEN_STATES) --use_proton=True --name=$(PROTON_NAME) --n_runs_benchmark=10 --n_runs_warmup=0 --bench_fn=own 2>&1 | tee $(PROTON_RESULTS_DIR)/$(PROTON_NAME)/speed-test.txt
	proton-viewer -m num_samples/%,time/s kernel.hatchet -i timing > $(PROTON_RESULTS_DIR)/$(PROTON_NAME)/kernel-by-line.txt
	mv kernel.hatchet $(PROTON_RESULTS_DIR)/$(PROTON_NAME)/kernel.hatchet

COMPARISON_DIR=profiles/barrier-comparison

barrier-comparison:
	@echo "=== Barrier version (working tree) ==="
	# Speed test
	mkdir -p $(COMPARISON_DIR)/barrier
	$(PYTHON) ./speed_test.py --name helion --n_hidden_states=$(N_HIDDEN_STATES) \
		--n_runs_warmup=25 --n_runs_benchmark=100 \
		--tgt_dir $(COMPARISON_DIR)/barrier 2>&1 | tee $(COMPARISON_DIR)/barrier/speed-test.txt
	# Proton
	mkdir -p $(COMPARISON_DIR)/barrier/proton
	$(PYTHON) ./speed_test.py --name helion --n_hidden_states=$(N_HIDDEN_STATES) \
		--use_proton=True --n_runs_benchmark=20 --n_runs_warmup=5 2>&1 | tee $(COMPARISON_DIR)/barrier/proton/speed-test.txt
	proton-viewer -m num_samples/%,time/s kernel.hatchet -i timing > $(COMPARISON_DIR)/barrier/proton/kernel-by-line.txt
	mv kernel.hatchet $(COMPARISON_DIR)/barrier/proton/kernel.hatchet
	# NCU (inlined to survive git stash)
	mkdir -p $(COMPARISON_DIR)/barrier/ncu
	$(SUDO) $(NCU) --set full --nvtx --nvtx-include "kernel/" \
		--kernel-name _helion_fused_sample_helion_kernel --import-source yes \
		-fo $(COMPARISON_DIR)/barrier/ncu/helion \
		$(PYTHON) ./speed_test.py --name helion --n_hidden_states $(N_HIDDEN_STATES) --n_runs_benchmark $(N_RUNS_BENCHMARK)
	@echo "=== Stash to two-stage version ==="
	git stash
	# Speed test
	mkdir -p $(COMPARISON_DIR)/two-stage
	$(PYTHON) ./speed_test.py --name helion --n_hidden_states=$(N_HIDDEN_STATES) \
		--n_runs_warmup=25 --n_runs_benchmark=100 \
		--tgt_dir $(COMPARISON_DIR)/two-stage 2>&1 | tee $(COMPARISON_DIR)/two-stage/speed-test.txt
	# Proton
	mkdir -p $(COMPARISON_DIR)/two-stage/proton
	$(PYTHON) ./speed_test.py --name helion --n_hidden_states=$(N_HIDDEN_STATES) \
		--use_proton=True --n_runs_benchmark=20 --n_runs_warmup=5 2>&1 | tee $(COMPARISON_DIR)/two-stage/proton/speed-test.txt
	proton-viewer -m num_samples/%,time/s kernel.hatchet -i timing > $(COMPARISON_DIR)/two-stage/proton/kernel-by-line.txt
	mv kernel.hatchet $(COMPARISON_DIR)/two-stage/proton/kernel.hatchet
	# NCU (inlined to survive git stash)
	mkdir -p $(COMPARISON_DIR)/two-stage/ncu
	$(SUDO) $(NCU) --set full --nvtx --nvtx-include "kernel/" \
		--kernel-name _helion_fused_sample_helion_kernel --import-source yes \
		-fo $(COMPARISON_DIR)/two-stage/ncu/helion \
		$(PYTHON) ./speed_test.py --name helion --n_hidden_states $(N_HIDDEN_STATES) --n_runs_benchmark $(N_RUNS_BENCHMARK)
	@echo "=== Restore barrier version ==="
	git stash pop