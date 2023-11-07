#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=<account>
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH -c 28
#SBATCH --mem=12000
#SBATCH --array=1-10
#SBATCH --job-name="train-autogluon"
#SBATCH --output=idun_out/test-srun.out
#SBATCH --mail-user=<email>
#SBATCH --mail-type=ALL
 
WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}

module purge
module load Python/3.10.8-GCCcore-12.2.0
pip install -r requirements.txt 

python train.py