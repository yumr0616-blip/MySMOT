"""重依赖区:需要 torch / transformers / opencv / PIL 的全部实现。

核心包(smot/ 顶层与 smot/datasets/)保持 stdlib-only,绝不 import 本
子包;依赖方向永远是 smot.ml -> smot,不允许反向。安装方式:

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    pip install -e .[ml]

模块一览:
    frames.py          帧提供者(图像目录/视频文件)+ 画框 grounding
    qwen_adapter.py    冻结 Qwen3.5 的 MLLMAdapter 实现 + soft token 注入
    unary_kfa.py       可学习 Unary KFA(Stage-1a)
    pairwise_kfa.py    可学习 Pairwise KFA(Stage-1b)
    fact_selector.py   可学习 Fact Selector(Stage-1b)
    projector.py       可学习 Projector(Stage-1a 起;1b 扩槽位)
    checkpoint.py      checkpoint 保存/加载(权重+构造配置+统计量,1a/1b 两种格式)
    training.py        训练循环(教师强制;Stage-1b 四模块联合,见模块内 docstring)
    gradient_check.py  验收门禁 #1:梯度恰好落在可训练槽位(python -m smot.ml.gradient_check)
"""
