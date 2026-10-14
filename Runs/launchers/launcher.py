import argparse
import yaml
import shutil
import subprocess
import os
from datetime import datetime
from itertools import product
import time
import threading


def _fmt_mb(value):
    """Format an MB value; show N/A when None."""
    return f"{value:.2f}" if value is not None else "N/A"


class ResourceMonitor:
    """Background thread that periodically samples the memory and GPU
    memory usage of the subprocess tree and records peak values.

    - RAM: total RSS of the whole process tree (prefers psutil,
      falls back to parsing /proc when psutil is unavailable)
    - GPU memory: prefers pynvml, falls back to calling nvidia-smi;
      when gpu_ids is None, all GPUs are monitored and the max is taken
    - Sampling interval is `interval` seconds; periodic sampling may
      miss very short transient spikes
    """

    def __init__(self, pid, gpu_ids=None, interval=1.0):
        self.pid = pid
        self.gpu_ids = gpu_ids
        self.interval = interval
        self.peak_rss_mb = None
        self.peak_gpu_mb = None
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        """Stop sampling and return (peak RAM MB, peak GPU MB);
        None if nothing was sampled. Safe to call multiple times."""
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join()
        return self.peak_rss_mb, self.peak_gpu_mb

    def _run(self):
        while not self._stop_event.is_set():
            try:
                rss = self._sample_rss_mb()
                if rss is not None:
                    self.peak_rss_mb = max(self.peak_rss_mb or 0.0, rss)
                gpu = self._sample_gpu_mb()
                if gpu is not None:
                    self.peak_gpu_mb = max(self.peak_gpu_mb or 0.0, gpu)
            except Exception:
                pass  # monitoring errors must never affect the main flow
            self._stop_event.wait(self.interval)

    # ---------------- RAM sampling ----------------
    def _sample_rss_mb(self):
        try:
            import psutil
        except ImportError:
            return self._proc_tree_rss_mb()
        try:
            root = psutil.Process(self.pid)
            total = 0
            for p in [root] + root.children(recursive=True):
                try:
                    total += p.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            return total / (1024 ** 2)
        except psutil.NoSuchProcess:
            return None

    def _proc_tree_rss_mb(self):
        """Fallback without psutil: sum the process-tree RSS via /proc (MB)."""
        try:
            ppid_map = {}
            for name in os.listdir("/proc"):
                if not name.isdigit():
                    continue
                try:
                    with open(f"/proc/{name}/stat", "r") as f:
                        data = f.read()
                    rparen = data.rfind(")")
                    ppid_map[int(name)] = int(data[rparen + 2:].split()[1])
                except (OSError, ValueError, IndexError):
                    continue
            tree = {self.pid}
            changed = True
            while changed:
                changed = False
                for pid, ppid in ppid_map.items():
                    if ppid in tree and pid not in tree:
                        tree.add(pid)
                        changed = True
            total_kb = 0
            for pid in tree:
                try:
                    with open(f"/proc/{pid}/status", "r") as f:
                        for line in f:
                            if line.startswith("VmRSS:"):
                                total_kb += int(line.split()[1])
                                break
                except OSError:
                    continue
            return total_kb / 1024.0 if total_kb > 0 else None
        except OSError:
            return None

    # ---------------- GPU memory sampling ----------------
    def _sample_gpu_mb(self):
        used = self._gpu_used_list()
        if not used:
            return None
        if self.gpu_ids is not None:
            vals = [used[i] for i in self.gpu_ids if 0 <= i < len(used)]
        else:
            vals = used
        return max(vals) if vals else None

    @staticmethod
    def _gpu_used_list():
        """Return a list of used GPU memory (MB) per device;
        None when no GPU environment is available."""
        try:
            import pynvml
            pynvml.nvmlInit()
            try:
                res = []
                for i in range(pynvml.nvmlDeviceGetCount()):
                    h = pynvml.nvmlDeviceGetHandleByIndex(i)
                    res.append(pynvml.nvmlDeviceGetMemoryInfo(h).used / (1024 ** 2))
                return res
            finally:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                text=True, stderr=subprocess.DEVNULL,
            )
            return [float(x) for x in out.strip().splitlines() if x.strip()]
        except (OSError, subprocess.CalledProcessError, ValueError):
            return None


def get_output_dir(params):
    """Return the run's output directory from a params dict.

    Different benchmark scripts name the output-dir argument differently;
    currently both "save_dir" and "out_dir" are in use. Returns the first
    one found, or None when the entry defines neither.
    """
    for key in ("save_dir", "out_dir"):
        value = params.get(key)
        if value:
            return value
    return None


def build_command(env, script, params):
    
    cmd = ["conda", "run", "-n", env, "python", "-u", script]

    for k, v in params.items():
        if isinstance(v, list):
            cmd.append(f"--{k}")
            cmd += [str(x) for x in v]
        else:
            cmd += [f"--{k}", str(v)]

    return cmd


def run_method(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Resolve relative paths against the config file's directory,
    # so configs work regardless of the launcher's CWD.
    # Absolute paths are kept as-is (backward compatible).
    config_dir = os.path.dirname(os.path.abspath(config_path))

    env = cfg["env"]
    script = cfg["script"]
    if not os.path.isabs(script):
        script = os.path.normpath(os.path.join(config_dir, script))
    params = cfg["params"]

    work_dir = cfg.get("work_dir")
    if work_dir is None:
        work_dir = os.path.dirname(script)
    elif not os.path.isabs(work_dir):
        work_dir = os.path.normpath(os.path.join(config_dir, work_dir))

    params_copy = params.copy()
    data_list = params_copy.pop("data")

    param_sets = []
    for d in data_list:
        merged = {**params_copy, **d}
        # Resolve a relative output dir against the config file's directory,
        # so the launcher (log files) and the subprocess (--save_dir /
        # --out_dir) agree on the location. Absolute paths are kept as-is.
        for key in ("save_dir", "out_dir"):
            value = merged.get(key)
            if value and not os.path.isabs(value):
                merged[key] = os.path.normpath(os.path.join(config_dir, value))
        param_sets.append(merged)
    print(f"\nLoaded {len(param_sets)} runs from {config_path}")

    for i, param in enumerate(param_sets):
        print(f"\n--- Run {i+1}/{len(param_sets)} ---")

        cmd = build_command(env, script, param)
        print("Command:")
        print(" ".join(cmd))
        print(f"Work dir: {work_dir}")
        print(f"Params: {param}")

        # GPU control
        env_vars = os.environ.copy()
        device = param.get("device_id")
        if isinstance(device, list):
            env_vars["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in device)
        elif device is not None:
            env_vars["CUDA_VISIBLE_DEVICES"] = str(device)
        env_vars["PYTHONUNBUFFERED"] = "1"

        # GPU list for resource monitoring (None = monitor all GPUs, take max)
        if isinstance(device, list):
            gpu_ids = [int(x) for x in device]
        elif device is not None:
            gpu_ids = [int(device)]
        else:
            gpu_ids = None

        # Store run artifacts (run.log, config.yaml, params.yaml, time.txt)
        # directly in the data entry's output dir (save_dir or out_dir),
        # alongside the results written by the benchmark script. Fall back
        # to a timestamped directory under ./logs when the entry has neither.
        save_dir = get_output_dir(param)
        if save_dir:
            run_dir = save_dir
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_name = os.path.basename(config_path).replace(".yaml", "")
            dataset_name = param.get("name", f"run{i}")
            run_dir = f"logs/{log_name}_{dataset_name}_{timestamp}"
        os.makedirs(run_dir, exist_ok=True)
        log_file = os.path.join(run_dir, "run.log")
        print(f"Run dir: {run_dir}")
        # shutil.copy(config_path, os.path.join(run_dir, "config.yaml"))
        with open(os.path.join(run_dir, "config.yaml"), "w") as f:
            yaml.dump(cfg, f)
        with open(os.path.join(run_dir, "params.yaml"), "w") as f:
            yaml.dump(param, f)


        monitor = None
        peak_rss_mb = None
        peak_gpu_mb = None

        start_time = datetime.now()
        start_ts = time.time()
        try:
        
            with open(log_file, "w") as f:
                process = subprocess.Popen(
                    cmd,
                    cwd=work_dir,
                    env=env_vars,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                # Start resource monitoring (peak process-tree RAM + GPU memory)
                monitor = ResourceMonitor(process.pid, gpu_ids=gpu_ids)
                monitor.start()
                for line in process.stdout:
                    print(line, end="")
                    f.write(line)
                process.wait()
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(process.returncode, cmd)

            end_time = datetime.now()
            end_ts = time.time()
            duration_sec = end_ts - start_ts
            duration_min = duration_sec / 60
            peak_rss_mb, peak_gpu_mb = monitor.stop()
            time_file = os.path.join(run_dir, "time.txt")
            with open(time_file, "w") as f:
                f.write(f"dataset: {param.get('name', 'unknown')}\n")
                f.write(f"start_time: {start_time}\n")
                f.write(f"end_time: {end_time}\n")
                f.write(f"duration_seconds: {duration_sec:.2f}\n")
                f.write(f"duration_minutes: {duration_min:.2f}\n")
                f.write(f"peak_memory_mb: {_fmt_mb(peak_rss_mb)}\n")
                f.write(f"peak_gpu_memory_mb: {_fmt_mb(peak_gpu_mb)}\n")
            print(f"Finished. Log: {log_file}")
            print(f"Time: {duration_min:.2f} min")
            print(f"Peak memory: {_fmt_mb(peak_rss_mb)} MB, "
                  f"Peak GPU memory: {_fmt_mb(peak_gpu_mb)} MB")


        except subprocess.CalledProcessError as e:
            end_time = datetime.now()
            end_ts = time.time()

            duration_sec = end_ts - start_ts

            if monitor is not None:
                peak_rss_mb, peak_gpu_mb = monitor.stop()

            time_file = os.path.join(run_dir, "time.txt")
            with open(time_file, "w") as f:
                f.write(f"dataset: {param.get('name', 'unknown')}\n")
                f.write(f"status: FAILED\n")
                f.write(f"start_time: {start_time}\n")
                f.write(f"end_time: {end_time}\n")
                f.write(f"duration_seconds: {duration_sec:.2f}\n")
                f.write(f"peak_memory_mb: {_fmt_mb(peak_rss_mb)}\n")
                f.write(f"peak_gpu_memory_mb: {_fmt_mb(peak_gpu_mb)}\n")

            dataset_name = param.get("name", "unknown")
            print(f"Failed run {i} for {config_path}")
            print(f"Dataset: {dataset_name}")
            print(f"Check log: {log_file}")
            print(f"Peak memory: {_fmt_mb(peak_rss_mb)} MB, "
                  f"Peak GPU memory: {_fmt_mb(peak_gpu_mb)} MB")
            raise RuntimeError(
                f"\nRun failed!\n"
                f"Dataset: {dataset_name}\n"
                f"Config: {config_path}\n"
                f"Log file: {log_file}\n"
                f"Duration: {duration_sec:.2f}s\n"
            ) from e
        finally:
            # Safety net: make sure the monitor thread is always joined
            # (stop is idempotent)
            if monitor is not None:
                monitor.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmarks from yaml configs")
    parser.add_argument(
        "methods",
        nargs="+",
        help="One or more method names; config is loaded from ../configs/{name}.yaml",
    )
    args = parser.parse_args()

    for name in args.methods:
        config_path = f"../configs/{name}.yaml"
        print(f"\n===== Running {config_path} =====")
        run_method(config_path)