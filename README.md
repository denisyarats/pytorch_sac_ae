# SAC+AE implementaiton in PyTorch

## Requirements
We assume you have access to a gpu that can run CUDA 9.2. Then, the simplest way to install all required dependencies is to create an anaconda environment by running:
```
conda env create -f conda_env.yml
```
After the instalation ends you can activate your environment with:
```
source activate pytorch_sac_ae
```

## Instructions
To train an SAC+AE agent on the `cheetah run` task from image-based observations  run:
```
python train.py \
    --domain_name cheetah \
    --task_name run \
    --encoder_type pixel \
    --decoder_type pixel \
    --action_repeat 4 \
    --save_video \
    --save_tb \
    --work_dir ./log \
    --seed 1
```
This will produce a folder (`./save`) by default, where all the output is going to be stored including train/eval logs, tensorboard blobs, evaluation videos, and model snapshots. It is possible to attach tensorboard to a particular run using the following command:
```
tensorboard --logdir save
```
Then open up tensorboad in your browser.

You will also see some console output, something like this:
```
| train | E: 1 | S: 1000 | D: 0.8 s | R: 0.0000 | BR: 0.0000 | ALOSS: 0.0000 | CLOSS: 0.0000 | RLOSS: 0.0000
```
This line means:
```
train - training episode
E - total number of episodes 
S - total number of environment steps
D - duration in seconds to train 1 episode
R - episode reward
BR - average reward of sampled batch
ALOSS - average loss of actor
CLOSS - average loss of critic
RLOSS - average reconstruction loss (only if is trained from pixels and decoder)
```
These are just the most important number, more of all other metrics can be found in tensorboard.
Also, besides training, once in a while there is evaluation output, like this:
```
| eval | S: 0 | ER: 21.1676
```
Which just tells the expected reward `ER` evaluating current policy after `S` steps. Note that `ER` is average evaluation performance over `num_eval_episodes` episodes (usually 10).

## Running on the cluster
You can find the `run_cluster.sh` script file that allows you run training on the cluster. It is a simple bash script, that is super easy to modify. We usually run 10 different seeds for each configuration to get reliable results. For example to schedule 10 runs of `walker walk` simple do this:
```
./run_cluster.sh walker walk
```
This script will schedule 10 jobs and all the output will be stored under `./runs/walker_walk/{configuration_name}/seed_i`. The folder structure looks like this:
```
runs/
  walker_walk/
    sac_states/
      seed_1/
        id # slurm job id
        stdout # standard output of your job
        stderr # standard error of your jobs
        run.sh # starting script
        run.slrm # slurm script
        eval.log # log file for evaluation
        train.log # log file for training
        tb/ # folder that stores tensorboard output
        video/ # folder stores evaluation videos
          10000.mp4 # video of one episode after 10000 steps
      seed_2/
      ...
```
Again, you can attach tensorboard to a particular configuration, for example:
```
tensorboard --logdir runs/walker_walk/sac_states
```

For convinience, you can also use an iPython notebook to get aggregated over 10 seeds results. An example of such notebook is `runs.ipynb`


## Run entire testbed
Another scirpt that allow to run all 10 dm_control task on the cluster is here:
```
./run_all.sh
```
It will call `run_cluster.sh` for each task, so you only need to modify `run_cluster.sh` to change the hyper parameters.
