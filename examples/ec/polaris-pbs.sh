#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l filesystems=home:eagle
#PBS -l walltime=0:60:00
#PBS -q debug
#PBS -A FoundEpidem

NNODES=`wc -l < $PBS_NODEFILE`
NRANKS=1 # Number of MPI ranks to spawn per node
NDEPTH=64 # Number of hardware threads per rank (i.e. spacing between MPI ranks)
GPU_PER_NODE=4
NTOTRANKS=$(( NNODES * NRANKS ))

export http_proxy="http://proxy-01.pub.alcf.anl.gov:3128"
export https_proxy="http://proxy-01.pub.alcf.anl.gov:3128"
export ftp_proxy="http://proxy-01.pub.alcf.anl.gov:3128"

echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS}"

# Setup the deepspeed hostfile
cat $PBS_NODEFILE > hostfile 
sed -e 's/$/ slots=4/' -i hostfile

master_node=$(cat $PBS_NODEFILE| head -1)
export MASTER_ADDR=$(host $master_node | head -1 | awk '{print $4}')

# Load modules
module load conda/2023-10-04
conda activate evoforecast

# Set environment variables
export HF_HOME=/lus/eagle/projects/CVD-Mol-AI/braceal/cache/huggingface

# The path to the accelerate config file
accelerate_config_file=examples/ec/deepspeed_configs/deepspeed_ddp_dynamo_single_node.yaml

# Change to work directory
cd /lus/eagle/projects/CVD-Mol-AI/braceal/src/genslm-esm

# Check if a config file is provided as a command-line argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <config_file>"
    exit 1
fi

config_file="$1"

 #--main_process_ip ${MASTER_ADDR} \ #$(head -1 $PBS_NODEFILE) \
#mpiexec --np ${NTOTRANKS} -ppn ${NRANKS} -d ${NDEPTH} --cpu-bind depth accelerate launch \
accelerate launch \
--config_file ${accelerate_config_file} \
 --main_process_ip ${MASTER_ADDR} \
 --main_process_port 25900 \
 --num_machines ${NNODES} \
 --num_processes $((NTOTRANKS * GPU_PER_NODE)) \
 --deepspeed_hostfile hostfile \
 genslm_esm/train.py --config ${config_file}
