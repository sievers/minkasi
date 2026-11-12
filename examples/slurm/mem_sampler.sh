#!/bin/bash
# Log this node's used memory and the RSS of the user's python processes every INTERVAL seconds, until killed.
# Started once per node by a399_401.sbatch, so the peak is known even if the job is OOM-killed.
# Usage: mem_sampler.sh LOGDIR [INTERVAL]
log="$1/mem_$(hostname -s).log"
while true; do
    used=$(free -g | awk '/^Mem/{print $3}')
    rss=$(ps -u "$USER" -o rss=,comm= | awk '$2 ~ /python/ {n++; s += $1; if ($1 > m) m = $1}
        END {printf "%d python, max %.1f GB, sum %.1f GB", n, m / 1048576, s / 1048576}')
    echo "$(date +%F_%T) node used ${used} GB; $rss" >>"$log"
    sleep "${2:-20}"
done
