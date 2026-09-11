# 轮腿 RL 任务模板（大板块骨架）

> 一个可训练的任务 = **3 个文件 + 1 行注册**。
> 本模板只列大板块，不含具体实现——用来回答"一个任务主要由哪几块组成"。

## 全局图：三个文件各管什么

```
task 名（如 Robotics-Wheelbipe-V14-Flat-v0）
   │  一行注册（__init__.py）把下面三者绑在一起
   ├── env_cfg.py ──────────── "学什么"：机器人是谁、奖励怎么定（参数表，无逻辑）
   ├── env.py ───────────────── "怎么跑"：RL 循环每一环的实现（环境类）
   └── agents/rsl_rl_ppo_cfg.py "怎么学"：PPO 网络结构与超参
```

一次训练 step 的数据流（env.py 各板块的调用顺序）：

```
_get_observations ──► 策略网络 ──► _pre_physics_step / _apply_action
        ▲                                │
        │                          物理仿真推进
        │                                ▼
   _reset_idx ◄── _get_dones ◄──── _get_rewards
  (重置摔倒的)   (摔倒/超时?)      (权重 × 公式)
```


## 模板 1：env_cfg.py —— 参数表（只有配置，没有计算）

```python
# ① 导入：configclass / EventTerm / mdp 工具 / 机器人资产 / 父类配置

# ② 事件配置 EventCfg —— 域随机化（sim2real 的关键）
@configclass
class EventCfg(EventCfg父类):
    质量随机化   = EventTerm(func=..., mode="startup", params={...})   # 开局随机一次
    摩擦随机化   = EventTerm(func=..., mode="startup", params={...})
    增益随机化   = EventTerm(func=..., mode="reset",   params={...})   # 每次重置随机
    重置姿态扰动 = EventTerm(func=..., mode="reset",   params={...})

# ③ 课程配置 CurriculumCfg —— 可选：按训练表现逐步加难度/撤辅助
@configclass
class CurriculumCfg:
    某课程项 = CurrTerm(func=..., params={...})

# ④ 主配置类 —— 一个任务的全部"旋钮"
@configclass
class XXXEnvCfg(父类EnvCfg):
    # 4.1 机器人：用哪个资产模型
    robot_cfg = ...
    # 4.2 观测：噪声 / 延迟 / 历史长度 / 裁剪缩放 / 维度
    # 4.3 动作：延迟范围
    # 4.4 命令生成器 commands：速度指令采样范围、特殊训练模式
    # 4.5 ★奖励权重表 rewards = OrderedDict( 项名 = 权重, ... )   ← RL 的灵魂
    # 4.6 其余开关：状态机、地形、终止条件...

    def __post_init__(self):
        # 4.7 运行时修正：按开关组合调整维度/开关（保持参数自洽）
        ...

# ⑤ 变体类 —— 继承主配置，只覆盖差异项
@configclass
class XXXEnvCfg_Play(XXXEnvCfg): ...      # 演示版：关随机化/课程
@configclass
class XXXEnvCfg_Rough(XXXEnvCfg): ...     # 换地形版
```


## 模板 2：env.py —— 环境类（RL 循环的实现）

```python
# ① 导入：torch / isaaclab / 父类环境 / 本任务的 env_cfg

# ② 模块级辅助函数：读配置并补默认值、换算工具（可无）

# ③ 环境类：重写需要的钩子，其余逻辑全在父类里
class XXXEnv(父类Env):
    cfg: XXXEnvCfg                 # 指明本环境用的配置类

    def __init__(self, cfg, ...):
        super().__init__(...)      # 父类搭场景、建机器人/传感器
        # 初始化：找关节/连杆索引、建内部状态张量、自检部件数量

    def _apply_action(self):       # 把策略动作发给电机（腿/轮/云台）
        ...

    def _get_observations(self):   # 读传感器 → 加噪声/延迟 → 拼观测向量
        ...

    def _get_rewards(self):        # 各项奖励公式 × cfg.rewards 权重 → 求和
        ...

    def _get_dones(self):          # 返回 (是否摔倒终止, 是否超时)
        ...

    def _reset_idx(self, env_ids): # 摆回初始状态、清计数、重采指令
        ...

    # （可选）自定义板块：状态机、课程学习、数据录制、特殊模式控制
```


## 模板 3：agents/rsl_rl_ppo_cfg.py —— PPO 超参

```python
@configclass
class XXXPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # ① 训练循环：每轮步数 / 总迭代数 / 保存间隔 / 实验名
    # ② policy:   网络结构（actor/critic 隐层宽度、激活函数、初始噪声）
    # ③ algorithm:PPO 超参（learning_rate / clip_param / gamma / lam / 熵系数 ...）
```


## 收尾：__init__.py 里一行注册

```python
gym.register(
    id="Robotics-XXX-v0",                       # task 名
    entry_point="isaaclab.envs:ManagerBasedRLEnv",  # 或 direct 环境入口
    env_cfg_entry_point=XXXEnvCfg,              # ┐
    agent_cfg_entry_point="XXXPPORunnerCfg",    # ┘ 三个文件在此绑定
)
```

## 记忆口诀

| 板块 | 文件 | 一句话 |
| --- | --- | --- |
| 学什么 | env_cfg.py | 奖励权重表 + 域随机化 + 机器人/观测参数 |
| 怎么跑 | env.py | 观测→动作→奖励→结束→重置 五个钩子 |
| 怎么学 | agents/*ppo_cfg.py | 网络结构 + PPO 超参 |
| 绑定 | __init__.py | 一行 register 变成一个 task 名 |
