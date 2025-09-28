import os
import sys
import time
import datetime
import argparse
import subprocess
from pathlib import Path
import signal

# 全局变量，用于存储需要监控的子进程
child_processes = []

def cleanup(signum, frame):
    """信号处理函数，用于在脚本退出前清理子进程。"""
    print(f"\n🚦 捕获到信号 {signum}。正在清理后台训练进程...", flush=True)
    for p in child_processes:
        if p.poll() is None:  # 检查进程是否仍在运行
            try:
                # 强制杀死子进程的整个进程组，确保完全终止
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                print(f"🔪 已发送 SIGKILL 到 PID 为 {p.pid} 的进程组。", flush=True)
            except ProcessLookupError:
                pass  # 进程可能已经结束
    sys.exit(0)

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
                print(f"✅ 成功找到新的模型目录: {new_dir_name}", flush=True)
                return os.path.join(MODELS_BASE_DIR, new_dir_name)
        except FileNotFoundError:
            # 如果是第一次运行，基础模型目录可能还不存在
            pass
        time.sleep(5)
    print(f"❌ 等待新模型目录超时（{timeout}秒）。", flush=True)
    return None

def run_detached_command(command):
    """在后台运行一个完全分离的命令。"""
    print(f"🚀 正在执行命令:\n   {command}", flush=True)
    subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def launch_and_monitor_child(cmd_list, log_file):
    """
    启动一个需要被监控的子进程（例如训练脚本），并将其记录下来以便后续清理。
    """
    print(f"🔥 正在启动受监控的训练进程... 日志文件: {log_file}", flush=True)
    with open(log_file, 'wb') as f:
        # start_new_session=True 使子进程成为新会话的领导者，
        # 这使其能抵抗SIGHUP信号（类似于nohup），并创建一个新的进程组。
        p = subprocess.Popen(cmd_list, stdout=f, stderr=f, start_new_session=True)
    child_processes.append(p)
    print(f"   -> 训练进程已启动，PID: {p.pid}", flush=True)

def monitor_and_launch(model_run_dir, main_dir_name, folder_name, timeout_hours=24):
    """
    监控模型目录，并为每个新生成的模型启动评估和调试脚本。
    """
    print(f"👀 开始监控目录: {model_run_dir}", flush=True)

    # 创建用于存放日志和结果的子目录
    debug_dir = os.path.join(main_dir_name, "debug_marl_behavior")
    eval_dir = os.path.join(main_dir_name, "evaluation")
    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    
    processed_models = set()
    start_time = time.time()
    timeout_seconds = timeout_hours * 3600

    print("🕒 监控循环已启动，将为每个新模型自动触发评估与调试...", flush=True)

    while time.time() - start_time < timeout_seconds:
        try:
            # 查找所有以 _actor.keras 结尾的模型文件
            all_models = {f for f in os.listdir(model_run_dir) if f.endswith('_actor.keras')}
            new_models = all_models - processed_models

            if not new_models:
                time.sleep(30) # 如果没有新模型，等待30秒
                continue

            for model_file in sorted(list(new_models)): # 按名称排序以保证顺序
                print("\n" + "="*60, flush=True)
                print(f"⭐ 发现新模型: {model_file}", flush=True)
                
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
                run_detached_command(eval_cmd)

                # 启动 debug_marl_behavior.py (保持不变)
                debug_log = os.path.join(debug_dir, f'db_{base_name}.log')
                debug_cmd = (
                    f"nohup python debug_marl_behavior.py "
                    f"--model_path {model_path} > {debug_log} 2>&1 &"
                )
                run_detached_command(debug_cmd)
                
                processed_models.add(model_file)
                print(f"✅ 已为模型 '{model_file}' 触发评估和调试任务。", flush=True)
                print("="*60, flush=True)

        except FileNotFoundError:
            # 模型目录可能尚未被训练脚本创建
            time.sleep(10)
        except Exception as e:
            print(f"🔴 监控过程中发生错误: {e}", flush=True)
            time.sleep(60)

    print("🏁 监控结束（达到超时时间或脚本被中断）。", flush=True)

def launch_background_process(args):
    """
    作为启动器，创建目录和日志路径，并在后台重新启动脚本作为工作进程。
    """
    #print(f"✨ 自动化脚本启动器PID: {os.getpid()}", flush=True)

    # 1. 创建主目录
    now = datetime.datetime.now()
    safe_folder_name = args.folder_name.replace(" ", "_").replace("/", "-")
    main_dir_name = now.strftime('%m%d_%H%M') + '_' + safe_folder_name
    os.makedirs(main_dir_name, exist_ok=True)

    # 2. 定义日志文件路径 (使用固定、简洁的名称)
    log_file_name = "auto_train_monitor.log"
    log_file_path = os.path.join(main_dir_name, log_file_name)

    # 3. 构建在后台运行的命令
    # 使用 sys.executable 确保使用相同的Python解释器
    # 使用 -u 标志确保实时输出
    command_str = (
        f"nohup {sys.executable} -u {__file__} "
        f"\"{args.folder_name}\" "
        f"--internal-run "
        f"--main-dir \"{main_dir_name}\" "
        f"> \"{log_file_path}\" 2>&1 &"
    )

    print(f"🚀 正在后台启动自动化脚本...")
    subprocess.Popen(command_str, shell=True)
    time.sleep(1)  # 等待片刻以确保进程启动
    print(f"✅ 自动化流程已在后台开始。您可以关闭此终端。")
    print(f"📂 所有输出（包括此脚本的日志）将保存在: {main_dir_name}")
    print(f"📜 使用此命令查看实时日志: tail -f \"{log_file_path}\"")

def run_background_tasks(args):
    """
    作为后台工作进程，执行主要的训练和监控任务。
    """
    # 注册信号处理器，以便在被kill时能够清理子进程
    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup) # 处理 Ctrl+C

    main_dir_name = args.main_dir
    folder_name = args.folder_name
    safe_folder_name = folder_name.replace(" ", "_").replace("/", "-")

    print(f"✨ 自动化工作进程已启动，PID: {os.getpid()}", flush=True)
    print(f"📂 主运行目录: {main_dir_name}", flush=True)
    
    # 确保模型基础目录存在
    os.makedirs(MODELS_BASE_DIR, exist_ok=True)
    dirs_before = set(os.listdir(MODELS_BASE_DIR))

    # 启动训练 (使用包含时间戳和实验名的详细日志)
    now = datetime.datetime.now()
    train_log_name = f"{now.strftime('%m%d_%H%M%S')}_{safe_folder_name}.log"
    train_log = os.path.join(main_dir_name, train_log_name)
    train_cmd_list = ["python", "-u", "mappo/ppo_marl_train.py"]
    launch_and_monitor_child(train_cmd_list, train_log)
    
    time.sleep(10) 

    # 查找由训练脚本创建的新目录
    model_run_dir = find_new_model_dir(dirs_before)

    if model_run_dir:
        # 监控目录并启动其他脚本
        monitor_and_launch(model_run_dir, main_dir_name, folder_name)
    else:
        print("❌ 未能找到训练输出目录。正在中止监控。", flush=True)
        print(f"   请检查训练日志以获取错误信息: {train_log}", flush=True)

def main():
    """
    主函数，根据参数决定是作为启动器还是作为后台工作进程。
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
    # 添加内部参数，用户无需关心
    parser.add_argument(
        "--internal-run", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--main-dir", type=str, help=argparse.SUPPRESS
    )
    args = parser.parse_args()

    if args.internal_run:
        # 如果有内部运行标记，则执行后台任务
        run_background_tasks(args)
    else:
        # 否则，作为启动器，在后台重新启动自己
        launch_background_process(args)

if __name__ == "__main__":
    # 确保脚本从项目根目录运行
    if not os.path.exists('mappo/ppo_marl_train.py'):
        print("❌ 错误: 此脚本必须从 'MARL_FOR_W_Factory' 项目根目录运行。", flush=True)
        sys.exit(1)
    main()
