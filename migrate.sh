#!/usr/bin/env bash
# Copy this project to another machine over SSH (optionally through Tor) and set it up there.
#
#   ./migrate.sh user@host [options]
#   ./migrate.sh user@abcdef...xyz.onion --tor
#   (user@host:port and ssh://user@host:port are accepted too)
#
# Options:
#   --tor          route every SSH connection through Tor with torsocks (needed for .onion hosts)
#   --dest DIR     destination folder, relative to the remote home or absolute (default: finance/codebase)
#   --port N       SSH port (default 22, or the :port in the target)
#   --key FILE     private key to use (default: ssh's usual keys, e.g. ~/.ssh/id_ed25519)
#   --batch-mb N   send files in batches of about N MB (default 25; a dropped connection loses one batch)
#   --no-env       do not copy .env (API keys); setup creates an empty one from the template
#   --no-data      do not copy data/ (the download then starts over on the target)
#   --no-setup     only copy; skip creating the virtualenv and running the tests on the target
#   --force        copy into a destination that already holds a finished copy of the project
#   --dry-run      show what would happen, change nothing
#
# The copy is resumable: only files that are missing on the target or differ in size or
# modification time are sent, so after a dropped connection just run the same command again.
# Copies code, data/, results/ (incl. trial log) and .env; skips .venv, caches and job state,
# which are rebuilt on the target. Running jobs must be stopped first.
set -euo pipefail

SSH_BIN=${SSH_CMD:-ssh}                  # overrides for testing
TORSOCKS_BIN=${TORSOCKS_CMD:-torsocks}
DEST="finance/codebase"
PORT=""
KEY=""
BATCH_MB=25
TOR=0 COPY_ENV=1 COPY_DATA=1 SETUP=1 FORCE=0 DRY=0
TARGET=""
MARKER=".migrate_incomplete"

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit "${1:-0}"; }
die() { echo "ERROR: $*" >&2; exit 1; }
step() { echo; echo "==> $*"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tor) TOR=1; shift ;;
    --dest) DEST="${2:?--dest needs a value}"; shift 2 ;;
    --port) PORT="${2:?--port needs a value}"; shift 2 ;;
    --key) KEY="${2:?--key needs a value}"; shift 2 ;;
    --batch-mb) BATCH_MB="${2:?--batch-mb needs a value}"; shift 2 ;;
    --no-env) COPY_ENV=0; shift ;;
    --no-data) COPY_DATA=0; shift ;;
    --no-setup) SETUP=0; shift ;;
    --force) FORCE=1; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) usage 0 ;;
    -*) die "unknown option $1 (see --help)" ;;
    *) [[ -z "$TARGET" ]] || die "only one target allowed"; TARGET="$1"; shift ;;
  esac
done
[[ -n "$TARGET" ]] || usage 1
# ssh itself has no host:port syntax: split "ssh://user@host:port" / "user@host:port" here.
TARGET="${TARGET#ssh://}"
if [[ "$TARGET" =~ ^(.+):([0-9]+)$ && "$TARGET" != *"]"* ]]; then
  [[ -z "$PORT" || "$PORT" == "${BASH_REMATCH[2]}" ]] || die "port given twice: --port $PORT and :${BASH_REMATCH[2]}"
  TARGET="${BASH_REMATCH[1]}"; PORT="${BASH_REMATCH[2]}"
fi
PORT="${PORT:-22}"
[[ "$PORT" =~ ^[0-9]+$ ]] || die "invalid port $PORT"
[[ "$TARGET" != *.onion && "$TARGET" != *.onion:* ]] || [[ $TOR -eq 1 ]] || die "$TARGET is an onion address: add --tor"

cd "$(dirname "$(readlink -f "$0")")"
[[ -f ctl.py && -f config.py ]] || die "run this from the project folder"

# ---------------------------------------------------------------------------------------------
# SSH plumbing. One shared connection (ControlMaster) for every step, so a key passphrase is
# asked for at most once. With --tor every ssh process runs under torsocks, which also resolves
# the host name through Tor (no DNS leak, .onion works).
# ---------------------------------------------------------------------------------------------
SSH=("$SSH_BIN")
if [[ $TOR -eq 1 ]]; then
  command -v "$TORSOCKS_BIN" >/dev/null || die "torsocks not found (apt install torsocks)"
  SSH=("$TORSOCKS_BIN" "$SSH_BIN")
fi
CONNECT_TIMEOUT=$([[ $TOR -eq 1 ]] && echo 90 || echo 15)       # Tor circuits take a while
CTL_SOCK="${TMPDIR:-/tmp}/migrate-ssh-$$-%r@%h:%p"
SSH_OPTS=(-p "$PORT" -o ServerAliveInterval=30 -o ServerAliveCountMax=10
          -o ControlMaster=auto -o "ControlPath=$CTL_SOCK" -o ControlPersist=300)
if [[ -n "$KEY" ]]; then
  [[ -f "$KEY" ]] || die "key file $KEY not found"
  SSH_OPTS+=(-i "$KEY" -o IdentitiesOnly=yes)
fi
rsh() { "${SSH[@]}" "${SSH_OPTS[@]}" "$TARGET" "$@"; }
trap '"${SSH[@]}" "${SSH_OPTS[@]}" -O exit "$TARGET" >/dev/null 2>&1 || true' EXIT
QDEST=$(printf '%q' "$DEST")             # quoted for the remote shell; relative = remote home

# What gets copied: every file except the excluded ones, as "path<TAB>size<TAB>mtime" lines.
FIND_PRUNE="-path ./.venv -prune -o -path ./data/cache -prune -o -path ./results/run -prune \
  -o -name __pycache__ -prune -o -name .pytest_cache -prune"
[[ $COPY_DATA -eq 1 ]] || FIND_PRUNE+=" -o -path ./data -prune"
FIND_FILTER="-type f ! -name '*.parquet.tmp' ! -name $MARKER"
[[ $COPY_ENV -eq 1 ]] || FIND_FILTER+=" ! -path ./.env"
LIST_CMD="find . $FIND_PRUNE -o $FIND_FILTER -printf '%P\t%s\t%T@\n' | awk -F'\t' '{printf \"%s\t%s\t%d\n\", \$1, \$2, \$3}' | LC_ALL=C sort"
local_list() { bash -c "$LIST_CMD"; }
remote_list() { rsh "cd $QDEST 2>/dev/null && { $LIST_CMD; } || true"; }

# ---------------------------------------------------------------------------------------------
# 1. Nothing may be writing to the project while it is copied
# ---------------------------------------------------------------------------------------------
step "Checking for running jobs"
if pgrep -f "$(pwd)/.venv/bin/python .*(fetch_data|validate|backtest)\.py" >/dev/null \
   || pgrep -f "python (fetch_data|validate|backtest)\.py" >/dev/null; then
  die "a fetch/validate/backtest job is still running here. Stop it first: python3 ctl.py status / stop <job>"
fi
echo "none running"

# ---------------------------------------------------------------------------------------------
# 2. Login
# ---------------------------------------------------------------------------------------------
step "Checking SSH key login to $TARGET (port $PORT${KEY:+, key $KEY}$([[ $TOR -eq 1 ]] && echo ', via Tor'))"
if [[ $TOR -eq 1 ]]; then
  "$TORSOCKS_BIN" true 2>/dev/null || die "torsocks cannot start - is Tor running (systemctl status tor)?"
fi
if "${SSH[@]}" "${SSH_OPTS[@]}" -o BatchMode=yes -o ConnectTimeout="$CONNECT_TIMEOUT" "$TARGET" true 2>/dev/null; then
  echo "key login works"
else
  # Not usable without interaction: maybe the key has a passphrase (and no ssh-agent holds it).
  echo "non-interactive login failed; trying an interactive login (enter your key passphrase if asked)"
  ERRF=$(mktemp)
  if rsh -o ConnectTimeout="$CONNECT_TIMEOUT" true 2> >(tee "$ERRF" >&2); then
    echo "login works (connection is reused for the remaining steps)"
    rm -f "$ERRF"
  else
    sleep 0.2; ERR=$(cat "$ERRF"); rm -f "$ERRF"
    if grep -qiE "resolve|name resolution|Name or service not known" <<<"$ERR"; then
      die "cannot resolve $TARGET$([[ $TOR -eq 1 ]] && echo ' through Tor'). Check the address (no typos, a v3 onion address has 56 characters before .onion), that the onion service is online, and that Tor is connected (journalctl -u tor)."
    elif grep -qiE "refused|timed out|No route|unreachable|Network is|Connection closed|general SOCKS server failure|TTL expired" <<<"$ERR"; then
      die "cannot connect to $TARGET port $PORT. Check that sshd runs there and listens on that port$([[ $TOR -eq 1 ]] && echo ', that the onion service forwards it (HiddenServicePort), and retry - Tor connections sometimes fail on the first try')."
    elif grep -qiE "Host key verification failed|REMOTE HOST IDENTIFICATION" <<<"$ERR"; then
      die "host key problem for $TARGET (see the message above). Fix ~/.ssh/known_hosts, then rerun."
    elif ! grep -qiE "Permission denied" <<<"$ERR"; then
      die "login to $TARGET failed (see the ssh message above)."
    elif [[ $DRY -eq 1 ]]; then
      die "the server rejected the key. Set up key login first (ssh-copy-id), then rerun."
    else
      echo "the server rejected the key; running ssh-copy-id (asks for the account password once)"
      if [[ $TOR -eq 1 ]]; then
        "$TORSOCKS_BIN" ssh-copy-id -p "$PORT" ${KEY:+-i "$KEY"} "$TARGET"
      else
        ssh-copy-id -p "$PORT" ${KEY:+-i "$KEY"} "$TARGET"
      fi
    fi
  fi
fi

# ---------------------------------------------------------------------------------------------
# 3. Target requirements
# ---------------------------------------------------------------------------------------------
step "Inspecting the target"
INFO=$(rsh 'python3 -c "import sys; print(\"%d.%d\" % sys.version_info[:2])" 2>/dev/null || echo none;
            command -v tar >/dev/null && echo tar-ok || echo tar-missing;
            find --version 2>/dev/null | grep -q GNU && echo find-ok || echo find-other;
            nproc 2>/dev/null || echo "?";
            awk "/MemTotal/{printf \"%.1f\", \$2/1048576}" /proc/meminfo 2>/dev/null || echo "?"') \
  || die "cannot run commands on $TARGET"
read -r PYV TAROK FINDOK CPUS MEMGB <<<"$(echo $INFO)"
echo "python3 $PYV, $TAROK, $FINDOK, $CPUS CPUs, ${MEMGB} GB RAM"
[[ "$PYV" != none ]] || die "python3 is not installed on the target"
python3 - "$PYV" <<'EOF' || die "the target needs Python 3.10+ (has $PYV)"
import sys; maj, mi = map(int, sys.argv[1].split(".")); sys.exit(0 if (maj, mi) >= (3, 10) else 1)
EOF
[[ "$TAROK" == tar-ok ]] || die "tar is not installed on the target"
[[ "$FINDOK" == find-ok ]] || die "the target needs GNU find (standard on Linux)"

RESUMING=0
if rsh "test -e $QDEST/$MARKER"; then
  RESUMING=1
  echo "an earlier copy to $DEST was interrupted; resuming it"
elif rsh "test -e $QDEST/ctl.py"; then
  [[ $FORCE -eq 1 ]] || die "$DEST on $TARGET already holds a finished copy; use --force to update it (only changed files are sent; files not in this copy are kept)"
  echo "destination holds a finished copy; updating changed files (--force)"
fi

# ---------------------------------------------------------------------------------------------
# 4. Copy: only files missing on the target or differing in size / modification time
# ---------------------------------------------------------------------------------------------
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"; "${SSH[@]}" "${SSH_OPTS[@]}" -O exit "$TARGET" >/dev/null 2>&1 || true' EXIT
local_list > "$TMP/local"
remote_list > "$TMP/remote"
LC_ALL=C comm -23 "$TMP/local" "$TMP/remote" > "$TMP/todo"
N_ALL=$(wc -l < "$TMP/local")
N_TODO=$(wc -l < "$TMP/todo")
MB_TODO=$(awk -F'\t' '{s += $2} END {printf "%.0f", s / 1048576}' "$TMP/todo")
step "Copying to $TARGET:$DEST: $N_TODO of $N_ALL files to send (~${MB_TODO} MB)"
echo "env file: $([[ $COPY_ENV -eq 1 ]] && echo included || echo NOT copied); data: $([[ $COPY_DATA -eq 1 ]] && echo included || echo NOT copied)"

if [[ $DRY -eq 1 ]]; then
  echo "(dry run) would send these files in ~${BATCH_MB} MB batches, e.g.:"
  head -5 "$TMP/todo" | cut -f1 | sed 's/^/    /'
  [[ $N_TODO -le 5 ]] || echo "    ... and $((N_TODO - 5)) more"
elif [[ $N_TODO -gt 0 ]]; then
  rsh "mkdir -p $QDEST && touch $QDEST/$MARKER"
  # Split the work list into batches of about BATCH_MB, each sent as one tar stream.
  awk -F'\t' -v lim=$((BATCH_MB * 1048576)) -v dir="$TMP" '
    { if (n == 0 || acc + $2 > lim && acc > 0) { n++; acc = 0 } acc += $2; print $1 > (dir "/batch." n) }
    END { print n > (dir "/n_batches") }' "$TMP/todo"
  NB=$(cat "$TMP/n_batches")
  SENT_MB=0
  for ((b = 1; b <= NB; b++)); do
    f="$TMP/batch.$b"
    bmb=$(awk -F'\t' 'NR==FNR {sz[$1]=$2; next} {s += sz[$1]} END {printf "%.1f", s/1048576}' "$TMP/todo" "$f")
    for attempt in 1 2 3 4 5; do
      if tar czf - --verbatim-files-from -T "$f" | rsh "tar xzf - -C $QDEST"; then
        break
      fi
      [[ $attempt -lt 5 ]] || die "batch $b failed 5 times (connection unstable?). Rerun the same command to resume."
      echo "  batch $b failed (attempt $attempt); retrying in $((attempt * 15))s ..."
      sleep $((attempt * 15))
    done
    SENT_MB=$(awk -v a="$SENT_MB" -v b="$bmb" 'BEGIN {printf "%.1f", a + b}')
    echo "  batch $b/$NB done ($(wc -l < "$f") files, ${bmb} MB; ${SENT_MB}/${MB_TODO} MB total)"
  done
fi

if [[ $DRY -eq 0 ]]; then
  # Verify: the target must now have every file with identical size and mtime.
  remote_list > "$TMP/remote"
  MISSING=$(LC_ALL=C comm -23 "$TMP/local" "$TMP/remote" | wc -l)
  [[ $MISSING -eq 0 ]] || die "$MISSING file(s) still differ on the target. Rerun the same command to resume."
  rsh "rm -f $QDEST/$MARKER"
  [[ $COPY_ENV -eq 0 ]] || rsh "chmod 600 $QDEST/.env 2>/dev/null || true"
  echo "verified: all $N_ALL files present on the target with matching size and timestamp"
fi

# ---------------------------------------------------------------------------------------------
# 5. Setup
# ---------------------------------------------------------------------------------------------
if [[ $SETUP -eq 1 ]]; then
  step "Setting up on the target (virtualenv, requirements, tests)"
  if [[ $DRY -eq 1 ]]; then
    echo "(dry run) would run: cd $DEST && python3 ctl.py setup"
  else
    rsh "cd $QDEST && python3 ctl.py setup"
  fi
fi

step "Done"
SSHC="$([[ $TOR -eq 1 ]] && echo "torsocks ")ssh -p $PORT${KEY:+ -i $KEY}"
cat <<EOF
Control it from here with:
  $SSHC $TARGET "cd $DEST && python3 ctl.py status"
  $SSHC $TARGET "cd $DEST && python3 ctl.py start fetch"
  $SSHC $TARGET "cd $DEST && python3 ctl.py logs fetch -n 20"
Jobs keep running after you disconnect. On a machine with more than 8 GB RAM, raise
N_WORKERS / N_WORKERS_HEAVY in config.py ($CPUS CPUs, ${MEMGB} GB RAM on the target).
EOF
