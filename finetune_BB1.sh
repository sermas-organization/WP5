#!/bin/bash
#
#SBATCH --job-name=blenderbot_finetuning
#SBATCH --output=/storage/ukp/work/petrak/sermas/wp_5_1/job_output/slurm-%A.out
#SBATCH --error=/storage/ukp/work/petrak/sermas/wp_5_1/job_output/slurm-%A.err
#SBATCH --mail-user=petrak@ukp.informatik.tu-darmstadt.de
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1

module load cuda/11.1
source /ukp-storage-1/petrak/parlai_venv/bin/activate
srun parlai train_model --task multiwoz_v21 --model transformer/generator  --init-model zoo:blender/blender_90M/model --model-file /mnt/beegfs/work/petrak/sermas/wp_5_1/parlai/Blenderbot_90M/model --num-epochs 10 --batchsize 16 --dynamic-batching full --embedding-size 512 --n-layers 8 --ffn-size 2048 --dropout 0.1 --n-heads 16 --learn-positional-embeddings True --n-positions 512 --variant xlm --activation gelu --fp16 true --fp16-impl mem_efficient --text-truncate 512 --label-truncate 128 --dict-tokenizer bpe --dict-lower True -lr 1e-06 --optimizer adam --lr-scheduler reduceonplateau --gradient-clip 0.1 --validation-every-n-epochs 0.25 --betas 0.9,0.999 --update-freq 1 --skip-generation True --validation-patience 10 --save-after-valid True --validation-metric ppl --validation-metric-mode min --save-after-valid True --tensorboard_log True --tensorboard_logdir /mnt/beegfs/work/petrak/sermas/wp_5_1/tensorboard