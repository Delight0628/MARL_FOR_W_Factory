import os
import sys
import time
import datetime
import argparse
import subprocess
from pathlib import Path

# 模型的基础目录，相对于脚本位置
MODELS_BASE_DIR = "mappo/ppo_models"

def find_new_model_dir(dirs_before, timeout=120):
    """等待并返回在MODELS_BASE_DIR中新创建的目录路径。"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            current_dirs = set(os.listdir(MODELS_BASE_DIR))
            new_dirs = current_dirs - dirs_before
            if new_dirs:
                new_dir_name = new_dirs.pop()
                print(f"✅ 成功找到新的模型目录: {new_dir_name}")
                return os.path.join(MODELS_BASE_DIR, new_dir_name)
        except FileNotFoundError:
            # 如果是第一次运行，基础模型目录可能还不存在
            pass
        time.sleep(5)
    print(f"❌ 等待新模型目录超时（{timeout}秒）。")
    return None

def run_command(command):
    """使用nohup在后台运行一个命令并打印该命令。"""
    print(f"🚀 正在执行命令:\n   {command}")
    subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def monitor_and_launch(model_run_dir, main_dir_name, folder_name, timeout_hours=24):
    """
    监控模型目录，并为每个新生成的模型启动评估和调试脚本。
    """
    print(f"👀 开始监控目录: {model_run_dir}")

    # 创建用于存放日志和结果的子目录
    debug_dir = os.path.join(main_dir_name, "debug_marl_behavior")
    eval_dir = os.path.join(main_dir_name, "evaluation")
    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    
    processed_models = set()
    start_time = time.time()
    timeout_seconds = timeout_hours * 3600

    print("🕒 监控循环已启动，将为每个新模型自动触发评估与调试...")

    while time.time() - start_time < timeout_seconds:
        try:
            # 查找所有以 _actor.keras 结尾的模型文件
            all_models = {f for f in os.listdir(model_run_dir) if f.endswith('_actor.keras')}
            new_models = all_models - processed_models

            if not new_models:
                time.sleep(30) # 如果没有新模型，等待30秒
                continue

            for model_file in sorted(list(new_models)): # 按名称排序以保证顺序
                print("\n" + "="*60)
                print(f"⭐ 发现新模型: {model_file}")
                
                model_path = os.path.join(model_run_dir, model_file)
                base_name = model_file.replace('.keras', '')

                # 为当前模型创建一个专属的评估子目录
                marl_eval_subdir = os.path.join(eval_dir, f'ev_{base_name}')
                os.makedirs(marl_eval_subdir, exist_ok=True)
                
                # 将日志文件和输出都指向这个新目录
                eval_log = os.path.join(marl_eval_subdir, f'ev_{base_name}.log')
                eval_cmd = (
                    f"nohup python evaluation.py "
                    f"--model_path {model_path} "
                    f"--generalization --gantt "
                    f'--run_name "{folder_name}" '
                    f"--output_dir {marl_eval_subdir} > {eval_log} 2>&1 &"
                )
                run_command(eval_cmd)

                # 启动 debug_marl_behavior.py (保持不变)
                debug_log = os.path.join(debug_dir, f'db_{base_name}.log')
                debug_cmd = (
                    f"nohup python debug_marl_behavior.py "
                    f"--model_path {model_path} > {debug_log} 2>&1 &"
                )
                run_command(debug_cmd)
                
                processed_models.add(model_file)
                print(f"✅ 已为模型 '{model_file}' 触发评估和调试任务。")
                print("="*60)

        except FileNotFoundError:
            # 模型目录可能尚未被训练脚本创建
            time.sleep(10)
        except Exception as e:
            print(f"🔴 监控过程中发生错误: {e}")
            time.sleep(60)

    print("🏁 监控结束（达到超时时间或脚本被中断）。")


def main():
    """
    主函数，用于编排自动化训练流程。
    1. 创建一个运行目录。
    2. 启动训练过程。
    3. 监控新模型的生成并触发评估和调试。
    """
    parser = argparse.ArgumentParser(
        description="自动化MARL模型的训练、评估和调试流程。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "folder_name",
        type=str,
        help="为本次训练运行提供一个描述性名称 (例如, '更改奖励函数测试')。"
    )
    args = parser.parse_args()

    # 1. 设置目录和路径
    now = datetime.datetime.now()
    # 替换掉可能影响路径的特殊字符
    safe_folder_name = args.folder_name.replace(" ", "_").replace("/", "-")
    main_dir_name = now.strftime('%m%d_%H%M') + '_' + safe_folder_name
    os.makedirs(main_dir_name, exist_ok=True)
    print(f"📂 已创建主运行目录: {main_dir_name}")

    # 确保模型基础目录存在
    os.makedirs(MODELS_BASE_DIR, exist_ok=True)
    dirs_before = set(os.listdir(MODELS_BASE_DIR))

    # 2. 启动训练
    train_log = os.path.join(main_dir_name, f"{now.strftime('%m%d_%H%M%S')}_{safe_folder_name}.log")
    train_cmd = f"nohup python mappo/ppo_marl_train.py > {train_log} 2>&1 &"
    run_command(train_cmd)
    print(f"🔥 训练进程已在后台启动。日志文件: {train_log}")
    
    # 等待一段时间以便训练脚本启动并创建目录
    time.sleep(10) 

    # 3. 查找由训练脚本创建的新目录
    model_run_dir = find_new_model_dir(dirs_before)

    if model_run_dir:
        # 4. 监控目录并启动其他脚本
        monitor_and_launch(model_run_dir, main_dir_name, args.folder_name)
    else:
        print("❌ 未能找到训练输出目录。正在中止监控。")
        print(f"   请检查训练日志以获取错误信息: {train_log}")

if __name__ == "__main__":
    # 确保脚本从项目根目录运行
    if not os.path.exists('mappo/ppo_marl_train.py'):
        print("❌ 错误: 此脚本必须从 'MARL_FOR_W_Factory' 项目根目录运行。")
        sys.exit(1)
    main()
