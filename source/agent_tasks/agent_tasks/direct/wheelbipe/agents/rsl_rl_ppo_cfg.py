# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# See LICENSE for full license terms.
#
# Authors:
#     Zhang Zhirui <2231625449@qq.com>
#     Cui Yu       <ctty694@gmail.com>
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# 【文件总览·给初学者】本文件是 PPO 训练算法的超参数配置（"怎么学"，区别于 env_cfg 的"学什么"）。
# 每个任务在注册时会绑定 (环境配置类, 本文件里的 Runner 配置类) 一对。
# RslRlOnPolicyRunnerCfg = 训练循环(采集→更新→存档)的配置；
# RslRlPpoActorCriticCfg = 策略网络(actor 出动作)+价值网络(critic 打分)的结构；
# RslRlPpoAlgorithmCfg   = PPO 损失函数与优化器的参数。
# 不懂每个超参的含义没关系，先记住三个最重要的：
#   learning_rate（学习快慢）、clip_param（单次更新幅度上限）、entropy_coef（探索程度）。
# ─────────────────────────────────────────────────────────────────────────────

from isaaclab.utils import configclass  # 配置类装饰器

from isaaclab_rl.rsl_rl import (   # Isaac Lab 官方提供的 RSL-RL 配置基类
    RslRlOnPolicyRunnerCfg,        # 训练 runner 配置（迭代数、保存间隔、实验名…）
    RslRlPpoActorCriticCfg,        # 标准 PPO actor-critic 网络结构配置
    RslRlPpoAlgorithmCfg,          # 标准 PPO 算法超参配置
)






@configclass
class Wheelbipe25V3FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # 25 赛季 V3 机器人的标准 PPO 配置（V13/V14 都继承它做微调）
    num_steps_per_env = 24          # 每轮每环境采 24 步就更新一次网络（和 env_cfg 里外推轮次的 24 对应）
    max_iterations = 20000          # 最多训练 20000 轮迭代
    save_interval = 500             # 每 500 轮保存一次 checkpoint
    experiment_name = "wheelbipe25_v3_flat_direct"  # 实验名（决定 logs/ 下的目录名）
    empirical_normalization = False # 是否对观测做经验标准化（False=用 env_cfg 里的固定缩放即可）
    policy = RslRlPpoActorCriticCfg(   # —— 策略/价值网络结构 ——
        init_noise_std=1.0,            # 动作噪声初始标准差（越大前期探索越猛）
        actor_hidden_dims=[512, 256, 128],  # actor(出动作) MLP 三层隐层宽度
        critic_hidden_dims=[512, 256, 128], # critic(打分) MLP 三层隐层宽度
        activation="elu",              # 激活函数 ELU（比 ReLU 平滑，RL 常用）
    )
    algorithm = RslRlPpoAlgorithmCfg(  # —— PPO 算法超参 ——
        value_loss_coef=2.0,           # 价值损失权重（critic 学得有多用力）
        use_clipped_value_loss=True,   # 价值损失也做裁剪（防价值估计大幅跳变）
        clip_param=0.2,                # ★PPO 裁剪系数：单次策略更新幅度上限（PPO 的核心）
        entropy_coef=0.005,            # 熵奖励系数：鼓励探索（太小容易过早收敛）
        num_learning_epochs=5,         # 每批数据重读 5 遍
        num_mini_batches=4,            # 每轮数据切 4 个小批次（mini batch size = 环境数×24/4）
        learning_rate=1.0e-4,          # ★学习率
        schedule="adaptive",           # 学习率自适应：按 KL 散度自动升降
        gamma=0.99,                    # 折扣因子：未来奖励打 99 折（越接近 1 越有远见）
        lam=0.95,                      # GAE 的 λ：优势估计的偏差/方差折中
        desired_kl=0.01,               # 自适应学习率的目标 KL（新旧策略差异控制在 0.01）
        max_grad_norm=1.0,             # 梯度裁剪：梯度范数超 1 就缩（防训练爆炸）
    )
    # algorithm = RslRlPpoAlgorithmCfg(  # （注释掉：一组旧的超参备份）
    #     value_loss_coef=1.0,
    #     use_clipped_value_loss=True,
    #     clip_param=0.2,
    #     entropy_coef=0.01,
    #     num_learning_epochs=5,
    #     num_mini_batches=4,
    #     learning_rate=1.0e-3,
    #     schedule="adaptive",
    #     gamma=0.99,
    #     lam=0.95,
    #     desired_kl=0.01,
    #     max_grad_norm=1.0,
    # )



@configclass
class DreamWaqPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # DreamWaQ 算法的 PPO 配置：runner/网络/算法都要换成 DreamWaQ 专用类，
    # 所以这里用 dict + class_name 的方式"按名字"指定类（运行时由框架解析）。
    runner_class = "OnPolicyDreamWaqRunner"   # 指定训练循环类（普通 runner 的 DreamWaQ 变体）
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 500
    experiment_name = "dreamwaq_flat_direct"

    policy = dict(                    # DreamWaQ 的策略网络（ActorCriticDreamWaq）
        class_name="ActorCriticDreamWaq",   # 网络类名（按名字解析）
        init_noise_std = 1.0,
        activation = 'elu',
        # AdaBoot configuration (for DreamWaq paper reproduction)
        # available: "off" | "reward_cv" | "uncertainty" | "hybrid"
        # paper-style: p_boot = 1 - tanh(CV(R))
        adaboot_mode = "off",         # AdaBoot 自适应 bootstrapping 模式（off=关闭）
        # clamp of alpha / p_boot (safety)
        adaboot_min = 0.0,            # alpha/p_boot 下限
        adaboot_max = 1.0,            # 上限
        # only used in uncertainty / hybrid mode
        adaboot_temperature = 1.0,    # 温度参数（仅 uncertainty/hybrid 模式用）
        adaboot_bias = 0.0,           # 偏置项
        cenet_encoder_hidden_dims=[256, 128, 64],  # CENet 编码器隐层（把历史压成隐状态）
        cenet_decoder_hidden_dims=[64, 128, 256],  # CENet 解码器隐层（用隐状态"想象"特权信息）
    )
    algorithm = dict(                 # DreamWaQ 的 PPO 变体（PPODreamWaq）
        class_name="PPODreamWaq",
        value_loss_coef = 4.0,        # 价值损失权重
        use_clipped_value_loss = True,
        clip_param = 0.2,
        entropy_coef = 0.005,
        num_learning_epochs = 5,
        num_mini_batches = 4, # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-4, #5.e-4
        vae_learning_rate = 1.e-3,    # CENet(VAE 部分)单独的学习率（比策略大，学得快）
        num_adaptation_module_substeps = 1,  # 适应模块每次更新的子步数
        kl_weight = 1.0,              # KL 正则权重（估计器和策略的一致性约束）
        schedule = 'adaptive', # could be adaptive, fixed
        gamma = 0.99,
        lam = 0.95,
        desired_kl = 0.01,
        max_grad_norm = 1.,
        # AdaBoot reward-window statistics
        adaboot_reward_window_size = 1024,   # AdaBoot 的奖励滑窗长度
        # p_boot = 1 - tanh(adaboot_reward_cv_scale * CV + adaboot_reward_cv_offset)
        # smaller scale -> larger p_boot
        adaboot_reward_cv_scale = 0.25,      # 奖励变异系数的缩放
        adaboot_reward_cv_offset = 0.,
        adaboot_pboot_min = 0.0,
        adaboot_pboot_max = 1.0,
    )



@configclass
class HIMPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # HIMLoco(HIM) 算法的 PPO 配置：估计器用历史观测推断特权信息（替代地形编码器）
    runner_class = "OnPolicyHIMRunner"        # HIM 专用训练循环
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 500
    experiment_name = "him_flat_direct"

    policy = dict(                    # HIM 的策略网络（ActorCriticHIM）
        class_name="ActorCriticHIM",
        init_noise_std = 1.0,
        actor_hidden_dims = [512, 256, 128],   # 标准 MLP
        critic_hidden_dims = [512, 256, 128],
        activation = 'elu',
    )
    algorithm = dict(                 # HIM 的 PPO 变体（PPOHIM，含估计器联合训练）
        class_name="PPOHIM",
        value_loss_coef = 4.0,
        use_clipped_value_loss = True,
        clip_param = 0.2,
        entropy_coef = 0.005,
        num_learning_epochs = 5,
        num_mini_batches = 4, # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-4, #5.e-4
        schedule = 'adaptive', # could be adaptive, fixed
        gamma = 0.99,
        lam = 0.95,
        desired_kl = 0.01,
        max_grad_norm = 1.,
    )

@configclass
class NP3OPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # NP3O 算法的 PPO 配置：PPO + 安全约束(cost) + BarlowTwins 自监督历史编码
    runner_class = "OnConstraintPolicyRunner"  # 带约束处理的训练循环
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 500
    experiment_name = "np3o_flat_direct"

    policy = dict(                    # NP3O 的策略网络（ActorCriticBarlowTwins）
        class_name="ActorCriticBarlowTwins",
        init_noise_std = 1.0,
        continue_from_last_std = True,         # 噪声标准差接续上次值（续训时用）
        scan_encoder_dims = [128, 64, 32],     # 高度扫描编码器维度（本任务 scan=0，实际不用）
        actor_hidden_dims = [512, 256, 128],
        critic_hidden_dims = [512, 256, 128],
        #priv_encoder_dims = [64, 20],
        priv_encoder_dims = [],                # 特权编码器维度（空=线性）
        activation = 'elu', # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm',                   # （仅循环网络用，当前不用）
        # rnn_hidden_size = 512,
        # rnn_num_layers = 1,

        tanh_encoder_output = False,           # 编码器输出是否过 tanh
        num_costs = 1,                         # 约束条数（基类版 1 条，V14 专用版改成 5 条）
        teacher_act = True,                    # 用教师(旧策略)动作辅助（DAgger 式）
        imi_flag = True,                       # 打开模仿学习分支
        hist_encoder = False                   # 历史编码器开关
    )
    algorithm = dict(                 # NP3O 算法（含约束拉格朗日更新）
        class_name="NP3O",
        value_loss_coef = 4.0,
        use_clipped_value_loss = True,
        clip_param = 0.2,
        entropy_coef = 0.005,
        num_learning_epochs = 5,
        num_mini_batches = 4, # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-4, #5.e-4
        schedule = 'adaptive', # could be adaptive, fixed
        gamma = 0.99,
        lam = 0.95,
        desired_kl = 0.01,
        max_grad_norm = 1.,
        cost_value_loss_coef = 0.1,   # 约束价值函数的损失权重
        cost_viol_loss_coef = 0.1,    # 约束违反的损失权重
        dagger_update_freq = 20       # DAgger 教师策略每 20 轮更新一次
    )







@configclass
class WheelbipeV13FlatPPORunnerCfg(Wheelbipe25V3FlatPPORunnerCfg):
    # V13 平地任务：继承 25v3 配置，只改实验名和价值损失权重
    experiment_name = "wheelbipe_v13_flat_direct"

    # RND 需要 rnd_state 映射；rsl_rl 的 resolve_rnd_config 会据此拼 RND 输入维并在内部将 weight *= step_dt
    # obs_groups = {                       # （注释掉：RND 探索奖励的观测组映射）
    #     "policy": ["policy"],
    #     "critic": ["critic"],
    #     "rnd_state": ["policy"],
    # }

    algorithm = RslRlPpoAlgorithmCfg(  # 重写算法配置：价值损失权重 2.0→4.0
        value_loss_coef=4.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # rnd_cfg=RslRlRndCfg(            # （注释掉：RND 随机网络蒸馏探索奖励的配置）
        #     weight=1.0,
        #     weight_schedule=None,
        #     reward_normalization=False,
        #     state_normalization=False,
        #     learning_rate=1.0e-4,
        #     num_outputs=1,
        #     predictor_hidden_dims=[-1],
        #     target_hidden_dims=[-1],
        # ),
        # symmetry_cfg=RslRlSymmetryCfg(   # （注释掉：左右对称数据增强配置）
        #     use_data_augmentation=True,
        #     use_mirror_loss=False,
        #     data_augmentation_func=compute_symmetric_states,
        #     mirror_loss_coeff=0.0,
        # ),
    )







@configclass
class WheelbipeV13RoughPPORunnerCfg(Wheelbipe25V3FlatPPORunnerCfg):
    # V13 粗糙地形任务：完全继承 25v3 配置，只改实验名
    experiment_name = "wheelbipe_v13_rough_direct"

@configclass
class WheelbipeV14FlatPPORunnerCfg(WheelbipeV13FlatPPORunnerCfg):
    # ★V14 平地任务的 PPO 配置：网络改小（参数少训练快，轮腿任务不需要太大网络）
    experiment_name = "wheelbipe_v14_2_flat_direct"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        # actor_hidden_dims=[512, 256, 128],
        # critic_hidden_dims=[512, 256, 128],
        actor_hidden_dims=[256, 128, 64],   # actor 三层 [256,128,64]
        critic_hidden_dims=[256, 128, 64],  # critic 同样
        # actor_hidden_dims=[128, 64, 32],
        # critic_hidden_dims=[128, 64, 32],
        activation="elu",
    )

@configclass
class WheelbipeV14FlatDreamWaqPPORunnerCfg(DreamWaqPPORunnerCfg):
    # V14 平地 DreamWaQ：继承 DreamWaQ 配置，只改实验名和迭代数
    experiment_name = "wheelbipe_v14_flat_dreamwaq_direct"
    max_iterations = 10000          # 迭代减半（该算法收敛更快）



@configclass
class WheelbipeV14FlatHIMPPORunnerCfg(HIMPPORunnerCfg):
    # V14 平地 HIMLoco：同上
    experiment_name = "wheelbipe_v14_flat_him_direct"
    max_iterations = 5000           # HIM 迭代更少

@configclass
class WheelbipeV14FlatNP3OBarlowPPORunnerCfg(NP3OPPORunnerCfg):
    # V14 平地 NP3O+BarlowTwins：显式重写一遍网络/算法字典（5 条安全约束版）
    experiment_name = "wheelbipe_v14_flat_np3o_barlow_direct"
    max_iterations = 3000
    policy = dict(                    # 网络：与基类相同但 num_costs=5
        class_name="ActorCriticBarlowTwins",
        init_noise_std=1.0,
        continue_from_last_std=True,
        scan_encoder_dims=[128, 64, 32],
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        priv_encoder_dims=[],
        activation="elu",
        tanh_encoder_output=False,
        num_costs=5,                  # ★5 条安全约束（倾角/身高/角速度/力矩/关节速度）
        teacher_act=True,
        imi_flag=True,
        hist_encoder=False,
    )
    algorithm = dict(                 # 算法：同基类 NP3O
        class_name="NP3O",
        value_loss_coef=4.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        cost_value_loss_coef=0.1,
        cost_viol_loss_coef=0.1,
        dagger_update_freq=20,
    )

@configclass
class WheelbipeV14RoughPPORunnerCfg(WheelbipeV13RoughPPORunnerCfg):
    # ★V14 粗糙地形任务的 PPO 配置：网络同 V14 平地（改小版）
    experiment_name = "wheelbipe_v14_2_rough_direct"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        # actor_hidden_dims=[512, 256, 128],
        # critic_hidden_dims=[512, 256, 128],
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        # actor_hidden_dims=[128, 64, 32],
        # critic_hidden_dims=[128, 64, 32],
        activation="elu",
    )
    # algorithm = dict(               # （注释掉：一个叫 PPOTaction 的算法变体备份）
    #     class_name="PPOTaction",
    #     value_loss_coef=4.0,
    #     use_clipped_value_loss=True,
    #     clip_param=0.2,
    #     entropy_coef=0.005,
    #     num_learning_epochs=5,
    #     num_mini_batches=4,
    #     learning_rate=1.0e-4,
    #     schedule="adaptive",
    #     gamma=0.99,
    #     lam=0.95,
    #     desired_kl=0.01,
    #     max_grad_norm=1.0,
    # )
