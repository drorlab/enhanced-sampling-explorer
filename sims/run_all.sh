#!/usr/bin/env bash
# Launch the production runs in the background; logs in results/logs/.
# usage: bash run_all.sh md metad gamd smd reference we remd sams
cd "$(dirname "$0")"; source env.sh
mkdir -p ../results/logs
for m in "$@"; do
  m_script=$m
  case $m in
    md)        args="--ns 20" ;;
    metad)     args="--ns 20" ;;
    gamd)      args="--cmd-ns 2 --eq-ns 4 --prod-ns 14" ;;
    smd)       args="--n 32 --workers 8" ;;
    reference) args="--ns 3 --workers 10" ;;
    reflow)    args="--ns 3 --workers 6 --low"; m_script=reference ;;
    we)        args="--workers 24 --iters 3000" ;;
    we2d)      args="--workers 24 --iters 3000 --bins2d --name we2d"; m_script=we ;;
    remd)      args="" ;;
    sams)      args="" ;;
    samsT)     args="--mode temperature --iters 12000 --name samsT"; m_script=sams ;;
  esac
  nohup python run_$m_script.py $args > ../results/logs/$m.log 2>&1 &
  echo "$m pid $!"
done
