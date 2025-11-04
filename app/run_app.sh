#!/bin/bash
# W工厂智能调度应用启动脚本

echo "=================================="
echo "🏭 W工厂智能调度系统"
echo "=================================="
echo ""

# 检查是否安装了streamlit
if ! command -v streamlit &> /dev/null
then
    echo "❌ 未检测到streamlit，正在安装..."
    pip install streamlit -q
fi

echo "正在启动Web应用..."
echo ""

# 获取本机IP地址
LOCAL_IPS=$(hostname -I | tr ' ' '\n' | grep -v "^$")

echo "🌐 应用启动后，请在浏览器中访问以下地址："
echo "=================================="
echo "   本地访问："
echo "   ✅ http://localhost:8501"
echo "   ✅ http://127.0.0.1:8501"
echo ""
if [ ! -z "$LOCAL_IPS" ]; then
    echo "   远程访问（局域网/云服务器）："
    for ip in $LOCAL_IPS; do
        echo "   ✅ http://$ip:8501"
    done
fi
echo ""
echo "   ❌ 不要访问：http://0.0.0.0:8501"
echo "      (0.0.0.0 是监听地址，不是访问地址)"
echo "=================================="
echo ""
echo "正在启动服务器..."
echo ""

# 进入应用目录并启动
cd "$(dirname "$0")"

# 启动Streamlit（使用配置文件 .streamlit/config.toml）
# --server.headless true: 不显示默认的URL提示
streamlit run app_scheduler.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false

echo ""
echo "应用已关闭"
