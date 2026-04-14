# eeg_preproc_deploy.py
# Run locally: python eeg_preproc_deploy.py
# Deploys eeg_preproc.py to cluster and submits SLURM job.
#
# Usage:
#   python eeg_preproc_deploy.py                    # all missing
#   python eeg_preproc_deploy.py --subject sub-dmnelf009
#   python eeg_preproc_deploy.py --overwrite

import argparse
import time
from pathlib import Path
from utils import run_ssh, scp_to
from config import CLUSTER_BASE, SLURM_ACCOUNT, LOCAL_BASE

CLUSTER_PYTHON = "/home/cccbauer/.conda/envs/eeg_preproc/bin/python"
CLUSTER_SCRIPT = CLUSTER_BASE + "/scripts/eeg_preproc.py"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject",   type=str, default=None)
    parser.add_argument("--task",      type=str, default=None)
    parser.add_argument("--run",       type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    # ── 1. Deploy eeg_preproc.py to cluster ────────────────
    local_script = Path(__file__).parent / "eeg_preproc.py"
    if not local_script.exists():
        print("ERROR: eeg_preproc.py not found at " + str(local_script))
        return

    print("Deploying eeg_preproc.py to cluster...")
    scp_to(local_script, CLUSTER_SCRIPT, verbose=False)
    print("Deployed: " + CLUSTER_SCRIPT)

    # ── 2. Build run command ────────────────────────────────
    cmd = CLUSTER_PYTHON + " " + CLUSTER_SCRIPT

    if args.subject and args.task and args.run:
        cmd += " --subject " + args.subject
        cmd += " --task "    + args.task
        cmd += " --run "     + args.run
        job_name = "eeg_" + args.subject.replace("sub-", "") + "_" + args.task + "_" + args.run
    elif args.subject:
        cmd += " --subject " + args.subject
        job_name = "eeg_" + args.subject.replace("sub-", "")
    else:
        cmd += " --all"
        job_name = "eeg_preproc_all"

    if args.overwrite:
        cmd += " --overwrite"

    # ── 3. Build SLURM script ──────────────────────────────
    log_out = CLUSTER_BASE + "/logs/" + job_name + "_%j.out"
    log_err = CLUSTER_BASE + "/logs/" + job_name + "_%j.err"

    sbatch_lines = [
        "#!/bin/bash",
        "#SBATCH --job-name=" + job_name,
        "#SBATCH --output=" + log_out,
        "#SBATCH --error="  + log_err,
        "#SBATCH --partition=short",
        "#SBATCH --time=08:00:00",
        "#SBATCH --cpus-per-task=4",
        "#SBATCH --mem=32G",
        "#SBATCH --account=" + SLURM_ACCOUNT,
        "",
        cmd,
    ]

    sbatch_name = "eeg_preproc.sh"
    sbatch_path = LOCAL_BASE / "scripts" / sbatch_name
    with open(sbatch_path, "w") as f:
        f.write("\n".join(sbatch_lines))

    # ── 4. Deploy SLURM script ─────────────────────────────
    scp_to(sbatch_path,
           CLUSTER_BASE + "/scripts/" + sbatch_name,
           verbose=False)
    print("Deployed: " + sbatch_name)
    print("Command:  " + cmd)

    # ── 5. Submit ──────────────────────────────────────────
    print("\nSubmitting SLURM job...")
    result = run_ssh("sbatch " + CLUSTER_BASE + "/scripts/" + sbatch_name)
    job_id = ""
    for line in result.stdout.strip().split("\n"):
        if "Submitted" in line:
            job_id = line.strip().split()[-1]
            print("Job ID: " + job_id)

    # ── 6. Monitor ─────────────────────────────────────────
    if job_id:
        print("\nMonitoring job " + job_id + "  (Ctrl+C to stop watching)")
        print("-" * 55)
        try:
            while True:
                r = run_ssh(
                    "squeue -j " + job_id
                    + " --format=%.8i_%.8T_%.10M 2>/dev/null",
                    verbose=False
                )
                status = r.stdout.strip()
                if status and "JOBID" not in status.split("\n")[-1]:
                    print(status)
                else:
                    print("Job finished — checking log...")
                    log = run_ssh(
                        "tail -40 " + CLUSTER_BASE
                        + "/logs/" + job_name + "_" + job_id + ".out 2>/dev/null",
                        verbose=False
                    )
                    print(log.stdout)
                    break
                time.sleep(20)
        except KeyboardInterrupt:
            print("\nStopped watching. Check manually:")
            print("  squeue -j " + job_id)
            print("  tail -f " + CLUSTER_BASE
                  + "/logs/" + job_name + "_" + job_id + ".out")

if __name__ == "__main__":
    main()