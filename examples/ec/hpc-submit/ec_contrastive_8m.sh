#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l filesystems=home:eagle
#PBS -l walltime=0:60:00
#PBS -q debug
#PBS -A RL-fold

NNODES=`wc -l < $PBS_NODEFILE`
NRANKS=1 # Number of MPI ranks to spawn per node
NDEPTH=64 # Number of hardware threads per rank (i.e. spacing between MPI ranks)

NTOTRANKS=$(( NNODES * NRANKS ))

export http_proxy="http://proxy-01.pub.alcf.anl.gov:3128"
export https_proxy="http://proxy-01.pub.alcf.anl.gov:3128"
export ftp_proxy="http://proxy-01.pub.alcf.anl.gov:3128"

echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS}"

# Load modules
module load conda/2023-10-04
conda activate evoforecast

# Change to work directory
cd /lus/eagle/projects/CVD-Mol-AI/braceal/src/genslm-esm

mpiexec --np ${NTOTRANKS} -ppn ${NRANKS} -d ${NDEPTH} --cpu-bind depth accelerate launch  genslm_esm/train.py --config configs/ec-v1/ec_contrastive_8m.yaml