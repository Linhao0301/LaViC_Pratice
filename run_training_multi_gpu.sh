#!/bin/bash

# LaViC Multi-GPU Training Script V2
# Enhanced version with GPU memory check and disk space optimization
# 使用方法: ./run_training_multi_gpu_v2_no_emoji.sh

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== LaViC Multi-GPU Training V2 ===${NC}"
echo -e "${BLUE}Enhanced with GPU check & disk optimization${NC}"
echo "Starting at $(date)"

# 检查GPU数量
GPU_COUNT=$(nvidia-smi -L | wc -l)
echo -e "${YELLOW}Detected $GPU_COUNT GPUs${NC}"

if [ $GPU_COUNT -lt 2 ]; then
    echo -e "${RED}Warning: Less than 2 GPUs detected.${NC}"
    echo -e "${YELLOW}Multi-GPU training requires at least 2 GPUs.${NC}"
    read -p "Continue with single GPU? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting..."
        exit 1
    fi
    GPU_COUNT=1
fi



# 设置日志文件
LOG_FILE="/root/autodl-tmp/LaViC/logs/training_multi_gpu_v2_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="/root/autodl-tmp/LaViC/logs/training_multi_gpu_v2.pid"

echo -e "${YELLOW}Log file: $LOG_FILE${NC}"
echo -e "${YELLOW}PID file: $PID_FILE${NC}"

# 记录系统信息
echo "=== Multi-GPU Training V2 System Info ===" >> $LOG_FILE
echo "Start time: $(date)" >> $LOG_FILE
nvidia-smi >> $LOG_FILE 2>&1
df -h >> $LOG_FILE 2>&1
echo "====================================" >> $LOG_FILE

# 设置环境变量优化
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 切换到源码目录
cd /root/autodl-tmp/LaViC/src

# 设置临时目录（避免系统盘空间不足）
export TMPDIR="/root/autodl-tmp/tmp"
mkdir -p $TMPDIR
echo -e "${YELLOW}Using temp directory: $TMPDIR${NC}"
echo "Temp directory: $TMPDIR" >> $LOG_FILE

# 显示配置信息
echo -e "${BLUE}Training Configuration:${NC}"
echo -e "   • GPUs: $GPU_COUNT"
echo -e "   • Batch size per GPU: 2"
echo -e "   • Total effective batch: $((2 * GPU_COUNT))"
echo -e "   • Strategy: DDP"
echo -e "   • Precision: 16-mixed"
echo -e "   • Workers per GPU: 8"

# 记录PID
echo $$ > $PID_FILE

echo -e "${GREEN}Starting Enhanced Multi-GPU Training...${NC}"

# 运行多GPU训练V2
python -u knowledge_distillation_multi_gpu_v2.py \
  --model_name llava-hf/llava-v1.6-mistral-7b-hf \
  --train_data ../data/item2meta_train.json \
  --val_data ../data/item2meta_valid.jsonl \
  --train_images_dir ../data/train_images \
  --val_images_dir ../data/valid_images \
  --output_dir ./out_distilled_multi_gpu \
  --lr 5e-5 \
  --weight_decay 1e-5 \
  --num_epochs 1 \
  --batch_size 1 \
  --num_workers 8 \
  --devices $GPU_COUNT \
  --strategy ddp \
  --max_samples 5000 \
  2>&1 | tee -a $LOG_FILE

# 检查训练结果
TRAINING_EXIT_CODE=${PIPESTATUS[0]}

# 记录训练结束
echo "=== Training End $(date) ===" >> $LOG_FILE
echo "Exit code: $TRAINING_EXIT_CODE" >> $LOG_FILE

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}Multi-GPU Training V2 completed successfully!${NC}"
    echo -e "${BLUE}Training Results:${NC}"
    
    # 检查LoRA adapter
    if [ -d "./out_distilled_multi_gpu/vision_lora_adapter_best_multi_gpu" ]; then
        ADAPTER_SIZE=$(du -sh "./out_distilled_multi_gpu/vision_lora_adapter_best_multi_gpu" | cut -f1)
        echo -e "   • LoRA adapter: $ADAPTER_SIZE"
        echo -e "${GREEN}   • LoRA adapter saved successfully${NC}"
        
        # 显示adapter文件
        echo -e "${PURPLE}   • Adapter files:${NC}"
        ls -la "./out_distilled_multi_gpu/vision_lora_adapter_best_multi_gpu/" | sed 's/^/     /'
    else
        echo -e "${RED}   • LoRA adapter not found${NC}"
    fi
    
    # 显示验证指标
    LATEST_METRICS=$(find "./out_distilled_multi_gpu" -name "val_metrics_epoch_*.txt" | sort | tail -1)
    if [ -f "$LATEST_METRICS" ]; then
        echo -e "${BLUE}   • Final validation metrics:${NC}"
        sed 's/^/     /' "$LATEST_METRICS"
    fi
    
    # 显示磁盘使用情况
    echo -e "${BLUE}   • Disk usage:${NC}"
    du -sh "./out_distilled_multi_gpu" | sed 's/^/     /'
    
    echo -e "${GREEN}Multi-GPU Training V2 completed successfully!${NC}"
    
else
    echo -e "${RED}Training failed with exit code: $TRAINING_EXIT_CODE${NC}"
    echo -e "${YELLOW}Check the log file for details: $LOG_FILE${NC}"
    
    # 显示最后几行错误信息
    echo -e "${RED}Last error messages:${NC}"
    tail -20 "$LOG_FILE" | sed 's/^/   /'
fi

# 显示系统状态
echo -e "${BLUE}Final System Status:${NC}"
if command -v nvidia-smi &> /dev/null; then
    echo -e "   GPU Status:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
    while IFS=, read -r idx name mem_used mem_total util; do
        echo -e "     GPU $idx ($name): ${mem_used}MB/${mem_total}MB (${util}% util)"
    done
fi

echo -e "   Disk Space:"
df -h . | tail -1 | awk '{print "     Available: " $4 " (" $5 " used)"}'

echo -e "${YELLOW}Full log: $LOG_FILE${NC}"

# 清理PID文件
rm -f $PID_FILE

echo -e "${GREEN}=== Multi-GPU Training V2 Complete ===${NC}"
