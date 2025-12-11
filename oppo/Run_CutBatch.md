## 1.训练文件与训练参数
训练文件是 `train111.sh`
脚本前可能需要更改：
```python
export HYDRA_FULL_ERROR=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3,2
```
更改的训练参数：
```python
./verl/trainer/config/ppo_trainer.yaml -> _temp_dir
./train111.sh -> data.train_batch_size=256 
./train111.sh -> data.max_prompt_length=256 
./train111.sh -> data.max_response_length=512 
./train111.sh -> actor_rollout_ref.actor.optim.lr=5e-8
./train111.sh -> actor_rollout_ref.actor.ppo_mini_batch_size=64 
./train111.sh -> actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 
./train111.sh -> actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 
./train111.sh -> actor_rollout_ref.rollout.tensor_model_parallel_size=2 
./train111.sh -> actor_rollout_ref.rollout.n=4 
./train111.sh -> actor_rollout_ref.rollout.cut_n=3 
./train111.sh -> actor_rollout_ref.rollout.cut_keep_rate=0.7 
./train111.sh -> actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=10
./train111.sh -> trainer.n_gpus_per_node=2 
./train111.sh -> trainer.nnodes=1 
```
训练参数未更改的原始训练脚本位于：`./verl/examples/grpo_trainer/run_qwen2_5_vl-7b.sh`但是`cut_n``cut_keep_rate`都是新参数

## 2.模型下载与环境配置
1.在oppo文件夹下有可以直接执行的从huggingface下载的命令，`download.sh`, **注意修改路径**
2.在oppo文件夹下有可以直接安装的`environment.yaml`,运行`conda env create -f environment.yaml`里面的prefix是在91服务器上的位置。torch的版本是2.6.0+cu124，**在实际测试中，torch这样的基础包需要先行安装，flash-attention,black需要有前置安装包，但是导出yaml文件的时候是按照package首字母排序的，默认安装顺序也是按照首字母安装，因此可能需要将他们的安装顺序往后放**