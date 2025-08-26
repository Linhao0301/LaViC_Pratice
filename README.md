# LaViC: 视觉知识蒸馏训练指南

> **⚠️ 当前状态**: 本仓库目前只复现了第一步**视觉模块的LoRA训练**。第二步的推荐提示调优训练还未完成复现测试。

这是LaViC (Large Vision-Language Conversational Recommendation) 项目的视觉知识自蒸馏训练部分，使用LoRA技术对LLaVA-v1.6模型的视觉模块进行微调。

复现过程中发现原代码的爬取图片和视觉模块lora训练代码有一些问题（没有实现并行，loss计算出错等），做了一些debug

## 📋 复现条件

### 环境
- **显存**: 至少20G 
- **CUDA**: 12.1
- **Python**: 3.8+

### 库
```
PyTorch: 2.3.0+cu121
TorchVision: 0.18.0+cu121  
PyTorch Lightning: 2.5.3
Transformers: 4.55.4
PEFT: 0.17.1
Pillow: 10.3.0
```

## 🚀 快速开始

### 1. 环境安装
```bash
cd LaViC
pip install -r requirements.txt
```

### 2. 数据准备
确保以下数据文件存在：
```
data/
├── item2meta_train.json      # 训练集产品元数据
├── item2meta_valid.jsonl     # 验证集产品元数据
├── train_images/             # 训练图片目录
└── valid_images/             # 验证图片目录
```

### 3. 启动训练

#### 方式一：直接运行 (推荐)
```bash
chmod +x run_training.sh
./run_training.sh
```

#### 方式二：后台运行 (可以关闭终端)
```bash
nohup ./run_training.sh > /dev/null 2>&1 &
```

## 📈 训练监控

### 查看训练进度
```bash
# 实时监控
./monitor_training.sh

# 或直接查看日志
tail -f logs/training_*.log
```

## ⚙️ 训练参数配置

所有训练参数都在 `run_training.sh` 文件中配置，主要参数说明：

```bash
# 核心训练参数
--batch_size 1           # 批大小 (推荐1，适配20GB显存)
--num_epochs 1           # 训练轮数
--lr 5e-5               # 学习率
--max_samples 5000      # 限制样本数量 (测试用，删除此行使用全部数据)

# 数据路径
--train_data ../data/item2meta_train.json
--val_data ../data/item2meta_valid.jsonl  
--output_dir ./out_distilled
```

### 配置选项说明

| 参数 | 建议值 | 说明 |
|------|--------|------|
| `batch_size` | 1 | 全数据训练推荐值，适配20GB显存 |
| `max_samples` | 删除此行 | 测试时限制5000样本，全训练时删除 |
| `num_epochs` | 1-3 | 视觉蒸馏通常1个epoch足够 |

## 📊 训练性能参考

### 全量数据训练 (~81,000样本)
- **显卡**: RTX 4090
- **显存占用**: ~20GB
- **批大小**: batch_size=1  
- **预计时间**: 4-5小时
- **RTX 3090**: 约1.5倍时间 (6-7小时)

### 测试训练 (5,000样本)
- **预计时间**: 20-30分钟
- **适用于**: 快速验证环境和参数

## 📁 训练输出文件说明

训练完成后会产生以下文件：

### 🎯 核心输出 (保留)
```
src/out_distilled/
├── vision_lora_adapter_best/          # ✅ 训练好的LoRA权重 (14MB)
│   ├── adapter_model.safetensors      # LoRA参数文件
│   ├── adapter_config.json           # LoRA配置文件  
│   └── README.md                      # 模型说明文档
└── val_metrics_epoch_1.txt            # ✅ 验证指标记录
```

### 📝 训练日志
```
logs/
└── training_YYYYMMDD_HHMMSS.log       # ✅ 详细训练日志
```

### 🗑️ 可删除文件
```
src/lightning_logs/                    # ❌ PyTorch Lightning调试日志 (可删除)
└── version_*/                         # TensorBoard事件和超参数记录
```



### 关键指标
- **train_loss**: 训练损失，应逐步下降
- **val_loss**: 验证损失，越低越好 (目标 < 0.6)
- **val_perplexity**: 验证困惑度，越低越好 (目标 < 2.0)

## ✅ 验证训练成功

训练成功的标志：
1. ✅ 存在 `src/out_distilled/vision_lora_adapter_best/adapter_model.safetensors` (约14MB)
2. ✅ 验证损失 < 0.6，困惑度 < 2.0
3. ✅ 日志显示 "Best LoRA adapter saved."



## 训练中断续训问题
由于使用LoRA轻量化训练，建议重新开始训练而非断点续训。

## 📚 下一步

完成视觉模块训练后，生成的 `vision_lora_adapter_best/` 将用于第二阶段的推荐提示调优训练。

---

