# LaViC Reproduction

> **⚠️ Current Status**: This repository currently only reproduces the first step: **LoRA training of the vision module**. The second step, prompt-tuning for recommendation, has not yet been reproduced or tested.

This is the vision knowledge self-distillation training part of the LaViC (Large Vision-Language Conversational Recommendation) project, where the vision module of the LLaVA-v1.6 model is fine-tuned using LoRA technology.

During reproduction, some issues were found in the original code for crawling images and vision module LoRA training (no parallelism, incorrect loss calculation, etc. ), and debugging was performed. 

## Files Modified
- crawl_images.py：concurrent downloads, network optimizations
- knowledge_distillation.py：Ensures correct loss computation through complete LLaVA model forward pass；error handling, robustness improvements

## 9.1 New：mutil-gpu training achieve
- ./src/knowledge_distillation_multi_gpu.py and ./run_training_multi_gpu.sh
- using DDP stragety, not FSDP (better, but not achieve yet). DDP stragety is easier but cost more vram, 'cause DDP will load complete model on each gpu, which means each gpu still cost 20gb vram when you set batch-size=1). 
- 2 gpus bring 1.8x acclerate

## 📋 Reproduction Requirements

### Environment
- **VRAM**: At least 20G  
- **CUDA**: 12.1  
- **Python**: 3.8+  

### Libraries

```
PyTorch: 2.3.0+cu121
TorchVision: 0.18.0+cu121
PyTorch Lightning: 2.5.3
Transformers: 4.55.4
PEFT: 0.17.1
Pillow: 10.3.0
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
cd LaViC
pip install -r requirements.txt
```

### 2. Data Preparation
Ensure the following data files exist:
```
data/
├── item2meta_train.json      # Training set product metadata
├── item2meta_valid.jsonl     # Validation set product metadata
├── train_images/             # Training image directory
└── valid_images/             # Validation image directory
```

### 3. Start Training

#### Method 1: Direct Run (Recommended)
```bash
chmod +x run_training.sh
./run_training.sh
```

#### Method 2: Run in Background (Close terminal safely)
```bash
nohup ./run_training.sh > /dev/null 2>&1 &
```

## 📈 Training Monitoring

### Check Training Progress
```bash
# Real-time monitoring
./monitor_training.sh

# Or view logs directly
tail -f logs/training_*.log
```

## ⚙️ Training Parameter Configuration

All training parameters are configured in the `run_training.sh` file. Main parameters:

```bash
# Core training parameters
--batch_size 1           # Batch size (recommended 1, fits 20GB VRAM)
--num_epochs 1           # Number of epochs
--lr 5e-5                # Learning rate
--max_samples 5000       # Limit number of samples (for testing, remove this line to use full data)

# Data paths
--train_data ../data/item2meta_train.json
--val_data ../data/item2meta_valid.jsonl  
--output_dir ./out_distilled
```


## 📊 Training Performance Reference

### Full Dataset Training (~81,000 samples)
- **GPU**: RTX 4090
- **VRAM**: ~20GB
- **Batch_Size**: batch_size=1  
- **Expected time**: 4-5hours (if RTX 3090, maybe ~1.5× longer (6-7 hours))



## 📁 Training Output Files

After training, the following files will be generated:

### 🎯 Core Output (Keep)
```
src/out_distilled/
├── vision_lora_adapter_best/          # ✅ Trained LoRA weights (14MB)
│   ├── adapter_model.safetensors      # LoRA parameter file
│   ├── adapter_config.json            # LoRA config file  
│   └── README.md                      # Model description
└── val_metrics_epoch_1.txt            # ✅ Validation metrics record
```

### 📝 Training Logs
```
logs/
└── training_YYYYMMDD_HHMMSS.log       # ✅ Detailed training logs
```

### 🗑️ Removable Files
```
src/lightning_logs/                    # ❌ PyTorch Lightning debug logs (can be deleted)
└── version_*/                         # TensorBoard events and hyperparameter logs
```



### Key Metrics
- **train_loss**: Training loss, should decrease gradually
- **val_loss**: Validation loss, the lower the better (target < 0.6)
- **val_perplexity**: Validation perplexity, the lower the better (target < 2.0)

## ✅ Verifying Training Success

Signs of successful training:
1. ✅ File exists: `src/out_distilled/vision_lora_adapter_best/adapter_model.safetensors` (~ 14MB)
2. ✅ Validation loss < 0.6, perplexity < 2.0
3. ✅ Logs display "Best LoRA adapter saved."


## Training Interruption / Resumption
Since LoRA lightweight training is used, it is recommended to restart training rather than resuming from checkpoints.

## 📚 Next Steps

After completing vision module training, the generated `vision_lora_adapter_best/` will be used for the second stage: recommendation prompt-tuning training.

---

