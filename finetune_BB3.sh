#!/bin/bash
#
#SBATCH --job-name=blenderbot_finetuning
#SBATCH --output=/storage/ukp/work/petrak/sermas/wp_5_1/job_output/slurm-%A.out
#SBATCH --error=/storage/ukp/work/petrak/sermas/wp_5_1/job_output/slurm-%A.err
#SBATCH --mail-user=petrak@ukp.informatik.tu-darmstadt.de
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --mem-per-gpu=32gb
#SBATCH --gres=gpu:2

module load cuda/11.7
conda activate venv_parlai
#srun python print_cuda.py
srun parlai multiprocessing_train --task multiwoz_v21 --model projects.bb3.agents.r2c2_bb3_agent:BB3SubSearchAgent --force-skip-retrieval True --search-server none --truncate 1024 --model-file /mnt/beegfs/work/petrak/sermas/ParlAI/data/models/bb3/bb3_3B/model --num-epochs 10 --batchsize 1 --validation-every-n-epochs 0.25  --validation-patience 10 --save-after-valid True --save-after-valid True --tensorboard_log True --tensorboard_logdir /mnt/beegfs/work/petrak/sermas/wp_5_1/bb3/tensorboard --ddp-backend ddp
