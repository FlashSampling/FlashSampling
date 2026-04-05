"""Per-rank nsys wrapper for torchrun.

torchrun launches this script as each worker. It reads LOCAL_RANK from
the environment, wraps the actual speed_test.py invocation with its own
nsys instance, and writes a per-rank .nsys-rep file.

Usage:
    torchrun --nproc_per_node=2 nsys_wrapper.py \
        --report-dir /path/to/dir \
        --name fused-triton \
        -- \
        /opt/fmms/speed_test.py --name fused-triton ...
"""

import os
import subprocess
import sys
import warnings


def _gpu_numa_node(gpu_index: int) -> int | None:
    """Look up the NUMA node for a GPU from sysfs."""
    try:
        pci_bus = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    str(gpu_index),
                    "--query-gpu=pci.bus_id",
                    "--format=csv,noheader",
                ],
                text=True,
            )
            .strip()
            .lower()
        )
        # nvidia-smi may return "00000000:XX:YY.Z", sysfs uses "0000:XX:YY.Z"
        if pci_bus.startswith("00000000:"):
            pci_bus = pci_bus[4:]
        numa_path = f"/sys/bus/pci/devices/{pci_bus}/numa_node"
        node = int(open(numa_path).read().strip())
        return node if node >= 0 else None
    except Exception as e:
        warnings.warn(f"[gpu {gpu_index}] Failed to look up NUMA node: {e}", stacklevel=2)
        return None


def _numa_bind(rank: int) -> None:
    """Pin this process to the NUMA node of its GPU (CPU + memory)."""
    node = _gpu_numa_node(rank)
    if node is None:
        print(f"[rank {rank}] NUMA node unavailable, skipping binding", flush=True)
        return
    try:
        # Read which CPUs belong to this NUMA node
        cpus_path = f"/sys/devices/system/node/node{node}/cpulist"
        cpu_list_str = open(cpus_path).read().strip()
        # Parse "0-47,96-143" format into a set of CPU ids
        numa_cpus = set()
        for part in cpu_list_str.split(","):
            if "-" in part:
                lo, hi = part.split("-")
                numa_cpus.update(range(int(lo), int(hi) + 1))
            else:
                numa_cpus.add(int(part))
        # Intersect with current affinity (cgroup may restrict)
        allowed = os.sched_getaffinity(0)
        target = numa_cpus & allowed
        if not target:
            print(f"[rank {rank}] No CPUs in NUMA node {node} within cgroup, skipping", flush=True)
            return
        os.sched_setaffinity(0, target)
        print(f"[rank {rank}] NUMA-bound to node {node}, CPUs {sorted(target)}", flush=True)
    except Exception as e:
        print(f"[rank {rank}] NUMA binding failed: {e}", flush=True)


def _print_numa_info(rank: str) -> None:
    """Print NUMA/CPU topology before nsys captures stdout."""
    cpus = sorted(os.sched_getaffinity(0))
    print(f"[rank {rank}] CPUs (after bind): {cpus}", flush=True)
    if rank == "0":
        try:
            topo = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True).strip()
            print(f"GPU topology:\n{topo}", flush=True)
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            print(f"[rank {rank}] Failed to get GPU topology: {e}", flush=True)


def main():
    args = sys.argv[1:]
    sep = args.index("--")
    wrapper_args = args[:sep]
    child_args = args[sep + 1 :]

    report_dir = wrapper_args[wrapper_args.index("--report-dir") + 1]
    name = wrapper_args[wrapper_args.index("--name") + 1]
    rank = os.environ["LOCAL_RANK"]

    _numa_bind(int(rank))
    _print_numa_info(rank)
    report_path = f"{report_dir}/{name}-rank{rank}"

    cmd = [
        "nsys",
        "profile",
        "-o",
        report_path,
        "--force-overwrite=true",
        "--capture-range=cudaProfilerApi",
        "--cuda-memory-usage=true",
        sys.executable,
        *child_args,
    ]
    print(f"[rank {rank}] Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd)
    # nsys exits with 143 (SIGTERM) after cudaProfilerStop, which is expected.
    # Report 0 to torchrun so it doesn't kill the other rank mid-write.
    rc = 0 if result.returncode in (0, 143, -15) else result.returncode
    sys.exit(rc)


if __name__ == "__main__":
    main()
