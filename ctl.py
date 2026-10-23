#!/usr/bin/env python3
"""
Remote control for the backtester: run everything from an SSH session.

Jobs run detached (own session, survive logout), at low CPU priority, with logs in
results/logs/<job>.log and state in results/run/<job>.json. Uses only the standard
library, so it works before the virtualenv exists.

    python3 ctl.py setup                       create .venv, install requirements, run tests
    python3 ctl.py start fetch                 download / resume the Alpaca data
    python3 ctl.py start validate-synthetic    validate all on synthetic data
    python3 ctl.py start validate              validate all on real data
    python3 ctl.py start validate -- permutation --heavy-workers 3   any validate.py arguments
    python3 ctl.py start backtest [-- --no-sweep]
    python3 ctl.py status                      jobs, progress, memory, data coverage, reports
    python3 ctl.py logs validate [-n 50] [-f]  show / follow a job's log
    python3 ctl.py stop validate               stop a job (and its worker processes)
    python3 ctl.py report [--synthetic]        print the validation REPORT.md
    python3 ctl.py test                        run the test suite
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
PY = VENV / "bin" / "python"
LOG_DIR = ROOT / "results" / "logs"
RUN_DIR = ROOT / "results" / "run"

# job name -> argv after the venv python
JOBS = {
    "fetch": ["fetch_data.py"],
    "validate": ["validate.py", "all"],
    "validate-synthetic": ["validate.py", "all", "--synthetic"],
    "backtest": ["backtest.py"],
    "backtest-synthetic": ["backtest.py", "--synthetic"],
    "test": ["-m", "pytest", "-q"],
}
HEAVY = {"validate", "validate-synthetic", "backtest", "backtest-synthetic"}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _state_path(job: str) -> Path:
    return RUN_DIR / f"{job}.json"


def _read_state(job: str) -> dict | None:
    p = _state_path(job)
    return json.loads(p.read_text()) if p.exists() else None


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # a zombie or a reused pid is not our job: check the command line
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        return str(PY) in cmd or "python" in cmd
    except OSError:
        return True


def _running(job: str) -> dict | None:
    st = _read_state(job)
    return st if st and _alive(st["pid"]) else None


def _mem_available_mb() -> float:
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return float("nan")


def _fmt_age(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def _last_lines(path: Path, n: int) -> list[str]:
    if not path.exists():
        return []
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 64 * 1024))
        data = f.read().decode(errors="replace")
    return [ln for ln in data.splitlines() if ln.strip()][-n:]


def _need_venv() -> None:
    if not PY.exists():
        sys.exit("No virtualenv yet. Run: python3 ctl.py setup")


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def cmd_setup(args) -> None:
    if sys.version_info < (3, 10):
        sys.exit(f"Python 3.10+ required, this is {sys.version.split()[0]}")
    if not PY.exists():
        print(f"creating {VENV} ...")
        r = subprocess.run([sys.executable, "-m", "venv", str(VENV)])
        if r.returncode != 0 or not (VENV / "bin" / "pip").exists():
            # Debian/Ubuntu without python3-venv: create without pip, then bootstrap pip.
            print("ensurepip unavailable; bootstrapping pip with get-pip.py")
            shutil.rmtree(VENV, ignore_errors=True)
            subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(VENV)], check=True)
            gp = ROOT / "results" / "get-pip.py"
            gp.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", gp)
            subprocess.run([str(PY), str(gp), "-q"], check=True)
            gp.unlink()
    print("installing requirements ...")
    subprocess.run([str(PY), "-m", "pip", "install", "-q", "--upgrade", "pip"], check=True)
    subprocess.run([str(PY), "-m", "pip", "install", "-q", "-r", str(ROOT / "requirements.txt")], check=True)
    env = ROOT / ".env"
    if not env.exists():
        shutil.copy(ROOT / ".env.example", env)
        os.chmod(env, 0o600)
        print(f"created {env} from the template - fill in ALPACA_API_KEY / ALPACA_SECRET_KEY")
    else:
        os.chmod(env, 0o600)
    if not args.skip_tests:
        cmd_test(args)
    cpus = os.cpu_count() or 1
    mem = _mem_available_mb()
    print(f"\nsetup done. machine: {cpus} CPUs, {mem:,.0f} MB available.")
    print("Worker counts are capped automatically by free memory (MEM_RESERVE_MB in config.py);"
          " raise N_WORKERS / N_WORKERS_HEAVY there on a bigger machine.")


def cmd_test(args) -> None:
    _need_venv()
    r = subprocess.run([str(PY), "-m", "pytest", "-q"], cwd=ROOT)
    if r.returncode:
        sys.exit("tests FAILED")


def cmd_start(args) -> None:
    _need_venv()
    job = args.job
    if job not in JOBS:
        sys.exit(f"unknown job {job!r}; choose from {', '.join(JOBS)}")
    if (st := _running(job)):
        sys.exit(f"{job} is already running (pid {st['pid']}, started {st['started']}). "
                 f"`python3 ctl.py stop {job}` first.")
    busy = [j for j in HEAVY if j != job and _running(j)]
    if job in HEAVY and busy and not args.force:
        sys.exit(f"heavy job(s) already running: {', '.join(busy)}. Running two at once can exhaust memory; "
                 "add --force to start anyway.")
    extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra
    argv = JOBS[job] + extra
    if job == "validate" and extra and extra[0] != "all" and not extra[0].startswith("-"):
        argv = ["validate.py"] + extra            # e.g. `start validate -- permutation`
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / f"{job}.log"
    with open(log, "a") as f:
        f.write(f"\n===== ctl start {job}: {' '.join(argv)} at {dt.datetime.now():%Y-%m-%d %H:%M:%S} =====\n")
    nice = ["nice", "-n", str(args.nice)] if shutil.which("nice") else []
    env = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1")
    with open(log, "a") as out:
        proc = subprocess.Popen(nice + [str(PY), *argv], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True, env=env)
    state = {"job": job, "pid": proc.pid, "argv": argv, "started": f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}",
             "started_ts": time.time(), "log": str(log)}
    _state_path(job).write_text(json.dumps(state, indent=2))
    time.sleep(1.5)
    if proc.poll() is not None:
        print("\n".join(_last_lines(log, 15)))
        sys.exit(f"{job} exited immediately (code {proc.returncode}); see {log}")
    print(f"started {job} (pid {proc.pid}); it keeps running after you log out.\n"
          f"  follow:  python3 ctl.py logs {job} -f\n  status:  python3 ctl.py status\n"
          f"  stop:    python3 ctl.py stop {job}")


def cmd_stop(args) -> None:
    st = _running(args.job)
    if not st:
        print(f"{args.job} is not running")
        return
    pgid = os.getpgid(st["pid"])
    os.killpg(pgid, signal.SIGTERM)                 # whole session: main process + workers
    for _ in range(30):
        if not _alive(st["pid"]):
            break
        time.sleep(0.5)
    else:
        os.killpg(pgid, signal.SIGKILL)
    with open(st["log"], "a") as f:
        f.write(f"===== ctl stop {args.job} at {dt.datetime.now():%Y-%m-%d %H:%M:%S} =====\n")
    print(f"stopped {args.job}. Fetch and the permutation test resume where they left off on the next start.")


def cmd_logs(args) -> None:
    log = LOG_DIR / f"{args.job}.log"
    if not log.exists():
        sys.exit(f"no log for {args.job} yet ({log})")
    if args.follow:
        os.execvp("tail", ["tail", "-n", str(args.n), "-F", str(log)])
    print("\n".join(_last_lines(log, args.n)))


def _data_summary() -> str:
    man = ROOT / "data" / "manifest.json"
    if not man.exists():
        return "no data downloaded yet"
    m = json.loads(man.read_text())
    bars = sorted({k.rsplit("/", 1)[-1] for k in m if k.startswith("bars/")})
    news = sorted(k.rsplit("/", 1)[-1] for k in m if k.startswith("news/"))
    syms = len({k.split("/")[1] for k in m if k.startswith("bars/")})
    parts = [f"bars {bars[0]}..{bars[-1]} ({len(bars)} months, {syms} symbols)" if bars else "no bars",
             f"news {news[0]}..{news[-1]} ({len(news)} months)" if news else "no news"]
    return "; ".join(parts)


def _report_line(root: Path) -> str:
    rep = root / "REPORT.md"
    if not rep.exists():
        return "not generated yet"
    txt = rep.read_text()
    n_pass, n_fail = txt.count("**PASS**"), txt.count("**FAIL**")
    age = time.time() - rep.stat().st_mtime
    return f"{n_pass} PASS / {n_fail} FAIL, written {_fmt_age(age)} ago ({rep.relative_to(ROOT)})"


def cmd_status(args) -> None:
    print(f"machine: {os.cpu_count()} CPUs, {_mem_available_mb():,.0f} MB RAM available, "
          f"load {os.getloadavg()[0]:.2f}")
    print(f"data:    {_data_summary()}")
    print(f"reports: real {_report_line(ROOT / 'results' / 'validation')}")
    print(f"         synthetic {_report_line(ROOT / 'results' / 'validation_synthetic')}")
    print("\njobs:")
    any_job = False
    for job in JOBS:
        st = _read_state(job)
        if not st:
            continue
        any_job = True
        alive = _alive(st["pid"])
        last = _last_lines(Path(st["log"]), 1)
        state = f"RUNNING {_fmt_age(time.time() - st['started_ts'])}" if alive else "stopped/finished"
        print(f"  {job:20s} {state:22s} pid {st['pid']:<7} {' '.join(st['argv'])}")
        if last:
            print(f"  {'':20s} last: {last[0][:110]}")
    if not any_job:
        print("  none started yet")


def cmd_report(args) -> None:
    root = ROOT / "results" / ("validation_synthetic" if args.synthetic else "validation")
    rep = root / "REPORT.md"
    if not rep.exists():
        sys.exit(f"{rep} does not exist yet")
    print(rep.read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("setup", help="create .venv, install requirements, run tests")
    p.add_argument("--skip-tests", action="store_true")
    sub.add_parser("test", help="run the test suite")
    p = sub.add_parser("start", help="start a job detached")
    p.add_argument("job", choices=list(JOBS))
    p.add_argument("--nice", type=int, default=10, help="CPU priority (0 normal .. 19 lowest); default 10")
    p.add_argument("--force", action="store_true", help="allow a second heavy job at the same time")
    p.add_argument("extra", nargs=argparse.REMAINDER, help="arguments after -- go to the job's script")
    p = sub.add_parser("stop", help="stop a job and its workers")
    p.add_argument("job", choices=list(JOBS))
    p = sub.add_parser("logs", help="show a job's log")
    p.add_argument("job", choices=list(JOBS))
    p.add_argument("-n", type=int, default=40)
    p.add_argument("-f", "--follow", action="store_true")
    sub.add_parser("status", help="jobs, memory, data coverage, reports")
    p = sub.add_parser("report", help="print the validation report")
    p.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    {"setup": cmd_setup, "test": cmd_test, "start": cmd_start, "stop": cmd_stop, "logs": cmd_logs,
     "status": cmd_status, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
