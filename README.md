# 仓库各版本说明
## 分支说明
1.`main`:最早的分支，在verl框架的基础上实现了无监督的多数投票。
2.`verl_maj_cutbatch`:最早实现截断逻辑的版本，该版本的截断是按照固定`cut_keep_rate`截取拼接的。这一版本的`reward_manager`是`majority`
3.`cut_majority_finish`:依然是固定`cut_keep_rate`截断，`reward_manager`更换为`trunc`，加入一致性奖励
4.`distribution_trunc`:截断比例分为5点截断`[0.2,0.35,0.5,0.65,0.8]`，基础奖励变为**答案在总体截断答案+母轨迹答案中的频率**
5.`distribution_trunc_01`:将4中的“频率基础答案”变为“多数投票01”，多数投票gt的获取是通过rollout_n条母轨迹获得，奖励构成为“准确性+一致性+格式”
6.`beta_trunc`：将reward建模成“频率奖励+beta分布稳定性调整”



# Version1.0
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

# Version2.0
## 命令行增加内容
**原来的这些内容配置在config中的绝对路径，现在将他们用命令行显式化**
`ray_kwargs.ray_init._temp_dir=/data1/yyy25/ray_tmp`用途： 指定 Ray 框架的临时目录。

`trainer.validation_data_dir=/data1/yyy25/verl/verl_valida`用途： 指定验证数据的存储目录。

`trainer.rollout_data_dir=/data1/yyy25/verl/verl_rollout`用途： 指定 Rollout 数据的存储目录。

`trainer.cut_data_dir=/data1/yyy25/verl/cut_rollout`用途： 指定 CUT  机制相关数据的存储目录。

`actor_rollout_ref.rollout.cut_n=5`用途： 控制 CUT 机制相关的 Rollout 样本数。
## 参数增加和代码变化
1.新增`cut_ratio_list=[0.2,0.35,0.5,0.65,0.8]`五个截断位置
2.原verl代码中数据处理阶段的截断策略是'error'，即不截断，但是因为91 2*A100的显存原因，所有均设置了“right”右截断
3.奖励调整逻辑目前设置的较为简单(原始`rollout.n`条轨迹并未算入信息增益计算集合，但是算入最后的答案集合)：
- 情况1: 答案在整体中存在，但在 max_cut_sign 中不存在。这可能表明这是一个“难以找到”的答案，因此我们提升奖励。`if max_cut_sign_frequency == 0 and base_frequency > 0: final_reward += base_reward * boost_factor`
- 情况2: 答案在 max_cut_sign 中的频率高于整体频率。这表明该答案在信息增益最大的截断点处表现良好，因此我们提升奖励。`elif max_cut_sign_frequency > base_frequency and base_frequency > 0: final_reward += base_reward * boost_factor`
- 情况3: 答案在 max_cut_sign 中的频率低于整体频率。这表明该答案在信息增益最大的截断点处表现不佳，因此我们降低奖励。`elif max_cut_sign_frequency < base_frequency and max_cut_sign_frequency > 0: final_reward -= base_reward * penalty_factor`

# Version2.1
## 命令行变化内容
1.删除了rollout输出的文件夹，并于主循环中将输出设置为`False`，加快一丢丢训练速度。
2.将`tensor_parallel_size`变为`1`，意在解决可能发生的内存泄漏和`rayOOM`问题
3.`_balance_batch`在`version2.0`版本中因为reward计算需要顺序删除，在现版本调整了位置而恢复，增加GPU利用率
4.**8A_train.sh**为新的训练脚本，更改内容如上，可直接应用

