# 回归测试

在 Isaac Lab Python 环境中，从仓库根目录执行 `python -m pytest`。pytest 只收集本目录，不收集本地参考例程中的测试。

测试通过 CPU 张量、几何计算、模型导出和模拟夹具验证核心逻辑，无需启动 Isaac Sim。依赖见 `requirements/test.txt`；CI 使用独立的 CPU PyTorch。

仿真器场景加载、GPU 训练、场地回放及硬件验证需要单独执行，不能由此处的通过结果推断。
