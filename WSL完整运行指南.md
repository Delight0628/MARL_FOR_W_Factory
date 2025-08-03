# 🐧 WSL Ubuntu中运行Ray MARL训练完整指南

## 📋 前置条件检查

### 1. 确认WSL安装和版本
在Windows PowerShell (管理员模式) 中运行：
```powershell
# 检查WSL状态
wsl --list --verbose

# 确保使用WSL2 (更好的性能)
wsl --set-version Ubuntu 2

# 检查WSL版本
wsl --version
```

### 2. 更新WSL内存配置 (可选但推荐)
在Windows用户目录下创建 `.wslconfig` 文件：
```ini
# C:\Users\[用户名]\.wslconfig
[wsl2]
memory=4GB
processors=4
swap=2GB
```

## 🚀 在WSL中设置环境

### 3. 进入WSL Ubuntu
```powershell
# 从Windows PowerShell进入WSL
wsl -d Ubuntu

# 或者直接启动Ubuntu应用
```

### 4. 复制项目文件到WSL
有几种方法：

**方法A: 在WSL中直接访问Windows文件**
```bash
# WSL可以直接访问Windows文件系统
cd /mnt/d/MPU/毕业论文/MARL_FOR_W_Factory
```

**方法B: 复制到WSL文件系统 (推荐，性能更好)**
```bash
# 复制整个项目到WSL用户目录
cp -r /mnt/d/MPU/毕业论文/MARL_FOR_W_Factory ~/marl_project
cd ~/marl_project
```

### 5. 运行环境设置脚本
```bash
# 确保脚本有执行权限
chmod +x setup_wsl_env.sh

# 运行设置脚本
bash setup_wsl_env.sh
```

设置脚本会自动：
- 更新Ubuntu系统包
- 安装Python3和pip
- 创建虚拟环境 `marl_env`
- 安装所有必需的Python包

### 6. 激活Python环境
```bash
# 激活虚拟环境
source marl_env/bin/activate

# 验证安装
python3 --version
pip list | grep ray
```

## 🎯 运行训练

### 7. 启动WSL版本的Ray训练
```bash
# 运行WSL优化的训练脚本
python3 wsl_ray_marl_train.py
```

### 8. 监控训练进度
训练过程中您会看到：
```
🐧 W工厂多智能体强化学习训练 - WSL Ubuntu版本
======================================================================
环境: WSL Ubuntu
框架: Ray RLlib
算法: PPO (Proximal Policy Optimization)
多智能体: 策略共享MAPPO
======================================================================
系统信息 - platform: WSL
系统信息 - cpu_count: 8
系统信息 - memory_mb: 8192
🔧 WSL系统配置:
   CPU核心: 8 (使用: 6)
   内存: 8192MB (对象存储: 500MB)
🚀 初始化Ray (WSL模式)...
✅ Ray初始化成功
```

## 📊 查看结果

### 9. 训练完成后查看结果
```bash
# 查看结果目录
ls -la wsl_ray_results/

# 查看训练摘要
cat wsl_ray_results/wsl_ray_training_summary_*.json

# 查看详细训练日志
ls wsl_ray_results/w_factory_wsl_marl_*/
```

### 10. 从Windows访问WSL文件
在Windows文件资源管理器地址栏输入：
```
\\wsl$\Ubuntu\home\[用户名]\marl_project\wsl_ray_results
```

## 🔧 故障排除

### 常见问题和解决方案

**问题1: Ray初始化失败**
```bash
# 重启WSL
wsl --shutdown
wsl -d Ubuntu

# 清理Ray缓存
rm -rf /tmp/ray*
```

**问题2: 内存不足**
```bash
# 检查内存使用
free -h

# 增加WSL内存限制 (在Windows中编辑 .wslconfig)
```

**问题3: 依赖包安装失败**
```bash
# 更新pip
pip install --upgrade pip

# 重新安装Ray
pip uninstall ray
pip install ray[rllib]
```

**问题4: 训练速度慢**
```bash
# 检查CPU使用
htop

# 调整训练参数 (编辑 wsl_ray_marl_train.py)
# 减少 num_iterations 或 num_rollout_workers
```

## 📈 性能优化建议

### WSL性能调优
1. **使用WSL2**: 确保使用WSL2而不是WSL1
2. **文件位置**: 将项目文件放在WSL文件系统中 (`~/`) 而不是Windows文件系统 (`/mnt/`)
3. **内存配置**: 在 `.wslconfig` 中分配足够内存
4. **CPU配置**: 合理设置CPU核心数

### Ray配置优化
1. **Worker数量**: 根据CPU核心数调整
2. **内存分配**: 根据系统内存调整对象存储大小
3. **批次大小**: 根据内存情况调整训练批次

## 🎉 成功标志

训练成功完成时，您会看到：
```
🎉 WSL Ray RLlib训练完成！
======================================================================
⏱️  训练时间: 15.30 分钟
🏆 最佳平均奖励: 245.67
📊 训练轮数: 50
📁 最佳检查点: /home/user/marl_project/wsl_ray_results/...
📄 训练摘要已保存: wsl_ray_results/wsl_ray_training_summary_*.json

✅ 这是在WSL中运行的真正MARL训练！
✅ 使用Ray RLlib框架
✅ PPO/MAPPO算法
✅ 多智能体策略共享
✅ Linux原生性能
```

## 📚 后续步骤

1. **模型评估**: 使用训练好的模型进行推理
2. **结果分析**: 分析训练曲线和性能指标
3. **参数调优**: 根据结果调整超参数
4. **部署应用**: 将模型集成到实际应用中

---

💡 **提示**: WSL环境下Ray的性能通常比Windows原生环境好很多，这是推荐的运行方式！ 