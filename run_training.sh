#!/bin/bash

# LaViC Training Script with Enhanced Monitoring
# 使用方法: ./run_training.sh

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== LaViC Training Script ===${NC}"
echo "Starting training at $(date)"

# 创建必要目录
mkdir -p /root/autodl-tmp/LaViC/logs
mkdir -p /root/autodl-tmp/LaViC/src/out_distilled

# 设置日志文件
LOG_FILE="/root/autodl-tmp/LaViC/logs/training_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="/root/autodl-tmp/LaViC/logs/training.pid"

echo -e "${YELLOW}Log file: $LOG_FILE${NC}"
echo -e "${YELLOW}PID file: $PID_FILE${NC}"

# 记录系统信息
echo "=== System Info ===" >> $LOG_FILE
nvidia-smi >> $LOG_FILE 2>&1
echo "=== Training Start $(date) ===" >> $LOG_FILE

# 切换到源码目录
cd /root/autodl-tmp/LaViC/src

# 设置临时目录到有足够空间的分区，避免checkpoint保存时空间不足
export TMPDIR="/root/autodl-tmp/tmp"
mkdir -p $TMPDIR
echo -e "${YELLOW}Using temp directory: $TMPDIR${NC}"
echo "Temp directory: $TMPDIR" >> $LOG_FILE

# 运行训练
python -u knowledge_distillation.py \
  --model_name llava-hf/llava-v1.6-mistral-7b-hf \
  --train_data ../data/item2meta_train.json \
  --val_data ../data/item2meta_valid.jsonl \
  --train_images_dir ../data/train_images \
  --val_images_dir ../data/valid_images \
  --output_dir ./out_distilled \
  --lr 5e-5 \
  --weight_decay 1e-5 \
  --num_epochs 1 \
  --batch_size 1 \
  --max_samples 5000 \
  2>&1 | tee -a $LOG_FILE

# 记录训练结束
echo "=== Training End $(date) ===" >> $LOG_FILE
echo -e "${GREEN}Training completed! Check log: $LOG_FILE${NC}"

# 清理PID文件
rm -f $PID_FILE
