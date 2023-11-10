#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=ie-idi
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=3      # 2 CPU cores
#SBATCH --nodes=2              # 2 compute nodes
#SBATCH --ntasks-per-node=1    # 1 mpi process each node
#SBATCH --mem=12000            # 12GB - in megabytes
#SBATCH --job-name="autogluon-10-11-2023"
#SBATCH --output=idun_out/autogluon-10-11-2023.out
#SBATCH --mail-user=taheeraa@ntnu.no

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "We are running from this directory: $SLURM_SUBMIT_DIR"
echo "The name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"

module purge
module load Python/3.10.8-GCCcore-12.2.0
pip install -r requirements.txt 

python train.py