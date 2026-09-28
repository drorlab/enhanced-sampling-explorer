# source sims/env.sh  - puts the esdemo env on PATH (tleap needs AMBERHOME)
export PATH=/opt/micromamba/envs/esdemo/bin:$PATH
export AMBERHOME=/opt/micromamba/envs/esdemo
export OPENMM_CPU_THREADS=${OPENMM_CPU_THREADS:-1}
