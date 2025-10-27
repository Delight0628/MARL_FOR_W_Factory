#!/bin/bash
# W工厂智能调度应用启动脚本

echo "=================================="
echo "🏭 W工厂智能调度系统"
echo "=================================="
echo ""
echo "正在启动Web应用..."
echo ""

# 检查是否安装了streamlit
if ! command -v streamlit &> /dev/null
then
    echo "❌ 未检测到streamlit，正在安装..."
    pip install streamlit -q
fi

# 进入应用目录并启动
cd "$(dirname "$0")"
streamlit run app_scheduler.py --server.port 8501 --server.address 0.0.0.0

echo ""
echo "应用已关闭"
