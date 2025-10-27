import re
import pandas as pd
import argparse
from typing import List, Dict, Any
import os

def parse_log_file(log_path: str) -> List[Dict[str, Any]]:
    """
    解析PPO训练日志文件，提取每个回合的关键指标。

    Args:
        log_path: 日志文件的路径。

    Returns:
        一个包含每回合数据的字典列表。
    """
    with open(log_path, 'r', encoding='utf-8') as f:
        log_content = f.read()

    # 正则表达式，用于匹配每个回合的数据块
    episode_block_regex = re.compile(
        r"🔂 回合\s+(\d+)/\d+.*?\| 奖励: (.*?)\s*\| Actor损失: (.*?)\|.*?本轮用时: ([\d.]+)s.*?\n"
        r"📊 KPI - 总完工时间: ([\d.]+)min\s*\|\s*设备利用率: ([\d.]+)%\s*\|\s*延期时间: ([\d.]+)min\s*\|\s*完成零件数: (\d+)/(\d+).*?\n"
        r"🚥 回合评分: ([\d.]+)\s*\(全局最佳: ([\d.]+)\)\s*\(阶段最佳: ([\d.]+)\)(.*?)\n",
        re.DOTALL
    )

    # 匹配所有课程阶段切换的区块
    stage_change_regex = re.compile(
        r"📚 \[回合 (\d+)\] 🔄 课程学习阶段切换!\n\s+新阶段: (.*?)\n"
    )

    # 提取所有数据
    episodes_data = []
    
    # 解析课程阶段
    stages = {}
    last_stage_start = 1
    last_stage_name = "初始阶段"
    for match in stage_change_regex.finditer(log_content):
        start_episode = int(match.group(1))
        stage_name = match.group(2).strip()
        
        if last_stage_name:
            for i in range(last_stage_start, start_episode):
                stages[i] = last_stage_name
        
        last_stage_start = start_episode
        last_stage_name = stage_name
        
    # 为最后一个阶段补充信息
    total_episodes = 0
    try:
        all_episode_nums = [int(e[0]) for e in episode_block_regex.findall(log_content)]
        if all_episode_nums:
            total_episodes = max(all_episode_nums)
    except (ValueError, IndexError):
        pass # 如果找不到，则保持为0

    for i in range(last_stage_start, total_episodes + 1):
        stages[i] = last_stage_name

    # 解析每个回合的数据
    for match in episode_block_regex.finditer(log_content):
        (
            episode, reward, actor_loss, iter_time,
            makespan, utilization, tardiness, completed_parts, target_parts,
            score, best_global_score, best_stage_score, model_update_info
        ) = match.groups()

        episode_num = int(episode)
        
        # 清理并转换数据类型
        data_dict = {
            '课程阶段 (Stage)': stages.get(episode_num, "未知"),
            '回合 (Episode)': episode_num,
            '奖励 (Reward)': float(reward.strip()),
            'Actor损失 (Actor_Loss)': float(actor_loss.strip()),
            '总完工时间 (Makespan_min)': float(makespan.strip()),
            '设备利用率 (Utilization_%)': float(utilization.strip()),
            '延期时间 (Tardiness_min)': float(tardiness.strip()),
            '完成零件数 (Completed_Parts)': int(completed_parts.strip()),
            '目标零件数 (Target_Parts)': int(target_parts.strip()),
            '回合评分 (Score)': float(score.strip()),
            '全局最佳评分 (Best_Global_Score)': float(best_global_score.strip()),
            '阶段最佳评分 (Best_Stage_Score)': float(best_stage_score.strip()),
            '本轮用时 (Iteration_Time_s)': float(iter_time.strip()),
            '模型是否更新 (Model_Updated)': 1 if '✅' in model_update_info else 0
        }
        episodes_data.append(data_dict)

    return episodes_data

def main():
    """主函数，用于解析命令行参数并执行日志解析。"""
    parser = argparse.ArgumentParser(
        description="将PPO训练日志文件解析并转换为同名的Excel文件。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "log_file",
        type=str,
        help="输入的日志文件路径, 例如: 'GPUtraining_log_20250827_232144.log'"
    )
    args = parser.parse_args()

    # 从日志文件名自动生成输出的Excel文件名
    base_name = os.path.splitext(args.log_file)[0]
    output_excel = f"{base_name}.xlsx"

    print(f"正在解析日志文件: {args.log_file}...")
    try:
        data = parse_log_file(args.log_file)
        if not data:
            print("❌ 错误：在日志文件中没有找到任何有效的回合数据。请检查文件内容和格式。")
            return
            
        df = pd.DataFrame(data)
        
        print(f"成功解析 {len(df)} 条回合数据。")
        print(f"正在将数据写入Excel文件: {output_excel}...")
        
        df.to_excel(output_excel, index=False, engine='openpyxl')
        
        print(f"✅ 成功！数据已保存到: {output_excel}")
        
    except FileNotFoundError:
        print(f"❌ 错误：找不到文件 '{args.log_file}'。请检查路径是否正确。")
    except Exception as e:
        print(f"❌ 解析过程中发生未知错误: {e}")


if __name__ == '__main__':
    main()
