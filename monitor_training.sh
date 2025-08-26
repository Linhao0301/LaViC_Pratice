#!/bin/bash

# LaViC Training Monitor Script
# 使用方法: ./monitor_training.sh

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

LOG_DIR="/root/autodl-tmp/LaViC/logs"

echo -e "${GREEN}=== LaViC Training Monitor ===${NC}"

# 检查是否有日志文件
if [ ! -d "$LOG_DIR" ] || [ -z "$(ls -A $LOG_DIR/*.log 2>/dev/null)" ]; then
    echo -e "${RED}No training logs found in $LOG_DIR${NC}"
    exit 1
fi

# 获取最新的日志文件
LATEST_LOG=$(ls -t $LOG_DIR/training_*.log 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo -e "${RED}No training log files found${NC}"
    exit 1
fi

echo -e "${YELLOW}Latest log file: $LATEST_LOG${NC}"
echo -e "${YELLOW}Log created: $(stat -c %y $LATEST_LOG)${NC}"

# 显示菜单
while true; do
    echo -e "\n${BLUE}Choose monitoring option:${NC}"
    echo "1) Real-time tail (follow training progress)"
    echo "2) Show training summary" 
    echo "3) Show GPU usage"
    echo "4) Show recent errors/warnings"
    echo "5) Show loss progression"
    echo "6) Exit"
    
    read -p "Enter your choice [1-6]: " choice
    
    case $choice in
        1)
            echo -e "${GREEN}Following log in real-time (Ctrl+C to stop):${NC}"
            tail -f "$LATEST_LOG" | grep --color=auto -E "(Epoch|loss|WARNING|ERROR|INFO)"
            ;;
        2)
            echo -e "${GREEN}=== Training Summary ===${NC}"
            echo "Log file: $LATEST_LOG"
            echo "File size: $(du -h $LATEST_LOG | cut -f1)"
            echo "Lines: $(wc -l < $LATEST_LOG)"
            echo ""
            echo "Training start:"
            grep "Training Start" "$LATEST_LOG" || echo "Not found"
            echo ""
            echo "Current progress:"
            tail -1 "$LATEST_LOG"
            echo ""
            echo "Warnings count: $(grep -c WARNING $LATEST_LOG || echo 0)"
            echo "Errors count: $(grep -c ERROR $LATEST_LOG || echo 0)"
            ;;
        3)
            echo -e "${GREEN}=== Current GPU Usage ===${NC}"
            nvidia-smi
            ;;
        4)
            echo -e "${GREEN}=== Recent Warnings/Errors ===${NC}"
            grep -E "(WARNING|ERROR)" "$LATEST_LOG" | tail -20
            ;;
        5)
            echo -e "${GREEN}=== Loss Progression ===${NC}"
            grep "train_loss_step" "$LATEST_LOG" | tail -20 | grep -o "train_loss_step=[0-9.]*"
            ;;
        6)
            echo -e "${GREEN}Goodbye!${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid option. Please choose 1-6.${NC}"
            ;;
    esac
    
    echo -e "\n${YELLOW}Press Enter to continue...${NC}"
    read
done
