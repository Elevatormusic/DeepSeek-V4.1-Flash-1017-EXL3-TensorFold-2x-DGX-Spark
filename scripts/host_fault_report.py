#!/usr/bin/env python3
"""Read host fault facts. This report does not identify a fault cause."""
import json
from pathlib import Path
import re
import subprocess


def command(args):
    """Keep command failures separate from an empty result."""
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=30, check=False)
    except FileNotFoundError:
        return {"status": "unavailable", "reason": "tool missing"}
    except (OSError, subprocess.TimeoutExpired):
        return {"status": "unavailable", "reason": "command failed or timed out"}
    if result.returncode or result.stderr.strip():
        return {"status": "unavailable", "reason": "command failed or access is incomplete"}
    return {"status": "read", "text": result.stdout.strip()}


def private_text(text):
    """Remove common private values. A person must still review the report."""
    text = re.sub(r"/(home|Users)/[^/\s]+", r"/\1/<user>", text)
    text = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "<address>", text)
    text = re.sub(r"\b(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}\b", "<mac>", text)
    return re.sub(r"(?i)(?:secret|password|token|authorization|api[_-]?key)\b.*", "<private value removed>", text)


def kernel_events(boot):
    result = command(["journalctl", "-k", "-b", str(boot), "--no-pager",
                      "--no-hostname", "-o", "short-monotonic", "-n", "2000"])
    if result["status"] != "read":
        return result
    lines = [private_text(line) for line in result.pop("text").splitlines()
             if re.search(r"AER|CmpltTO|completion.*timeout|CMD_SYNC|SMMU|mlx5|NVRM", line, re.I)]
    result.update(status="events" if lines else "no matching events", lines=lines[-200:],
                  scope="last 2000 kernel journal lines; at most 200 matching lines")
    return result


def read_file(path):
    try:
        return {"status": "read", "value": path.read_text().strip()}
    except OSError:
        return {"status": "unavailable", "reason": "file missing or not readable"}


def collect(sysfs=Path("/sys")):
    firmware = {p.parent.name: read_file(p) for p in sorted((sysfs / "class/infiniband").glob("*/fw_ver"))}
    thermal = {p.name: {"type": read_file(p / "type"), "millidegrees_c": read_file(p / "temp")}
               for p in sorted((sysfs / "class/thermal").glob("thermal_zone*"))}
    return {
        "notice": "Read-only facts, not a cause or a hardware pass. Review all output for private data before sharing.",
        "kernel": command(["uname", "-r"]),
        "gpu_driver_and_thermal": command(["nvidia-smi", "--query-gpu=driver_version,temperature.gpu,power.draw",
                                           "--format=csv,noheader,nounits"]),
        "previous_boot_kernel": kernel_events(-1),
        "current_boot_kernel": kernel_events(0),
        "rdma_firmware": firmware or {"status": "unavailable", "reason": "no RDMA firmware paths"},
        "thermal_zones": thermal or {"status": "unavailable", "reason": "no thermal zone paths"},
    }


if __name__ == "__main__":
    print(json.dumps(collect(), indent=2))
