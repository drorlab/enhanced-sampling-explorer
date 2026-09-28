#!/usr/bin/env bash
# Boot script for the CPU demo VM: micromamba + the esdemo env, then a check that
# openmm can build a CPU context. Idempotent, so a spot restart just re-checks.
set -uo pipefail
LOG=/var/log/esdemo-startup.log
exec >>"$LOG" 2>&1
echo "=== startup $(date -Is) ==="
export HOME=/root MAMBA_ROOT_PREFIX=/opt/micromamba
ENVDIR=$MAMBA_ROOT_PREFIX/envs/esdemo
meta() { curl -sf -H 'Metadata-Flavor: Google' \
  "http://metadata.google.internal/computeMetadata/v1/instance/$1" 2>/dev/null; }

if [ ! -x "$ENVDIR/bin/python" ]; then
  export DEBIAN_FRONTEND=noninteractive
  command -v bzip2 >/dev/null || { apt-get update -qq && apt-get install -y -qq bzip2 rsync; }
  command -v micromamba >/dev/null || curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xj -C /usr/local bin/micromamba
  meta attributes/env-yml > /tmp/environment.yml
  micromamba create -y -f /tmp/environment.yml || { echo "FATAL: env create failed"; exit 1; }
  chmod -R a+rX /opt/micromamba
fi
"$ENVDIR/bin/python" -c "
import openmm, openmmtools, pymbar
s = openmm.System(); s.addParticle(1.0)
c = openmm.Context(s, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName('CPU'))
print('openmm', openmm.version.version, 'openmmtools', openmmtools.__version__, 'pymbar', pymbar.__version__, 'CPU ok')
" && touch /opt/ENV_READY
echo "=== startup complete $(date -Is) ==="
