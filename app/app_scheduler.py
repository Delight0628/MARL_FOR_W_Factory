"""
W工厂智能调度应用 - 基于MARL的生产调度系统
支持模型加载、订单配置和调度结果可视化
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import json
import uuid
from typing import Dict
from i18n import LANGUAGES, get_text
import gymnasium as gym  # 10-25-14-30 引入以识别MultiDiscrete动作空间

# 设备选择：默认使用可用GPU；若需强制CPU，请设置环境变量 FORCE_CPU=1
if os.environ.get('FORCE_CPU', '0') == '1':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽TensorFlow的INFO级别日志

import tensorflow as tf
try:
    # 为所有可见GPU开启按需显存，以减少与其他进程的冲突
    if os.environ.get('FORCE_CPU', '0') != '1':
        _gpus = tf.config.list_physical_devices('GPU')
        for _g in _gpus:
            tf.config.experimental.set_memory_growth(_g, True)
except Exception:
    pass

# 添加项目路径
app_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(app_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from environments.w_factory_env import WFactoryEnv
from environments.w_factory_config import (
    PRODUCT_ROUTES, WORKSTATIONS, SIMULATION_TIME,
    get_total_parts_count, calculate_episode_score, generate_random_orders
)
from mappo.sampling_utils import choose_parallel_actions_multihead

# ============================================================================
# TensorFlow 2.15.0 兼容：健壮的模型加载函数
# ============================================================================

def load_actor_model_robust(model_path: str):
    """
    健壮的模型加载函数 - TensorFlow 2.15.0 兼容版本
    支持多种加载策略：.h5 -> .keras -> weights+meta重建
    
    10-26-17-30 改进：兼容新的时间戳子目录结构和旧的扁平结构
    """
    import re
    
    base_path = model_path.replace('.keras', '').replace('.h5', '').replace('_actor', '')
    
    # 10-26-17-45 修正：智能路径解析，兼容两种目录结构
    # 结构1（新）: models/20251026_155337/1026_1527/1026_1527base_train_best （文件名保留时间戳）
    # 结构2（旧）: models/20251026_155337/1026_1527base_train_best
    search_paths = [base_path]
    
    # 尝试从路径中提取时间戳并构建可能的路径
    path_parts = base_path.split('/')
    for i, part in enumerate(path_parts):
        # 匹配形如 "1026_1527base_train_best" 的模式
        match = re.match(r'^(\d{4}_\d{4})(.+)$', part)
        if match:
            timestamp = match.group(1)
            full_filename = part  # 完整文件名（保留时间戳）
            
            # 构建新结构路径：在文件名前插入时间戳子目录
            dir_parts = path_parts[:i]
            new_path = '/'.join(dir_parts + [timestamp, full_filename])
            if new_path not in search_paths:
                search_paths.append(new_path)
            break
    
    # 同时检查是否已经是新结构，需要尝试旧结构
    if len(path_parts) >= 2:
        parent_dir = path_parts[-2]
        if parent_dir and re.match(r'^\d{4}_\d{4}$', parent_dir):
            # 当前是新结构，构建旧结构路径
            filename = path_parts[-1]
            old_path = '/'.join(path_parts[:-2] + [filename])
            if old_path not in search_paths:
                search_paths.append(old_path)
    
    # 策略1：优先尝试H5格式（最稳定）
    # 在所有可能的路径中搜索
    for search_base in search_paths:
        h5_path = f"{search_base}_actor.h5"
        if os.path.exists(h5_path):
            try:
                model = tf.keras.models.load_model(h5_path, compile=False)
                return model
            except Exception:
                pass
    
    # 如果原始路径是完整的.h5文件，也尝试加载
    if model_path.endswith('.h5') and os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            return model
        except Exception:
            pass
    
    # 策略2：从权重+元数据重建
    # 在所有可能的路径中搜索
    for search_base in search_paths:
        meta_path = f"{search_base}_meta.json"
        weights_path = f"{search_base}_actor_weights.h5"
        
        if os.path.exists(meta_path) and os.path.exists(weights_path):
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                
                # 重建模型架构
                from mappo.ppo_network import PPONetwork
                
                action_space_meta = meta['action_space']
                if action_space_meta['type'] == 'MultiDiscrete':
                    action_space = gym.spaces.MultiDiscrete(action_space_meta['nvec'])
                else:
                    action_space = gym.spaces.Discrete(action_space_meta['n'])
                
                network = PPONetwork(
                    state_dim=meta['state_dim'],
                    action_space=action_space,
                    lr=None,
                    global_state_dim=meta['global_state_dim'],
                    network_config=meta.get('network_config')
                )
                
                network.actor.load_weights(weights_path)
                return network.actor
                
            except Exception:
                pass
    
    # 策略3：尝试.keras格式（最后的手段）
    # 在所有可能的路径中搜索
    for search_base in search_paths:
        keras_path = f"{search_base}_actor.keras"
        if os.path.exists(keras_path):
            try:
                model = tf.keras.models.load_model(keras_path, compile=False)
                return model
            except Exception:
                pass
    
    # 如果原始路径是完整的.keras文件，也尝试加载
    if model_path.endswith('.keras') and os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            return model
        except Exception:
            pass
    
    return None

# ============================================================================
# 页面配置
# ============================================================================
def setup_page():
    """设置页面配置，只在开始时运行一次"""
    lang = get_language()
    st.set_page_config(
        page_title=get_text("page_title", lang),
        page_icon="🏭",
        layout="wide",
        initial_sidebar_state="auto"
    )
    
    # 隐藏右上角的Deploy按钮和菜单，并优化样式
    hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stDeployButton {display: none;}
        
        /* 移除顶部空白 */
        .block-container {
            padding-top: 2rem !important;
        }
        
        /* 增大一级标题字号 */
        h1 {
            font-size: 2.5rem !important;
            font-weight: 700 !important;
            margin-bottom: 1.5rem !important;
        }
        
        /* 二级标题字号 */
        h2 {
            font-size: 1.8rem !important;
            font-weight: 600 !important;
            margin-top: 2rem !important;
            margin-bottom: 1rem !important;
        }
        
        /* 三级标题字号 */
        h3 {
            font-size: 1.3rem !important;
            font-weight: 500 !important;
        }
        </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ============================================================================
# 辅助函数
# ============================================================================

def load_custom_products():
    """从文件加载自定义产品配置"""
    config_dir = os.path.join(app_dir, "custom_products")
    new_file = os.path.join(config_dir, "custom_products.json")
    old_file = os.path.join(app_dir, "custom_products.json")
    try:
        os.makedirs(config_dir, exist_ok=True)
    except Exception:
        pass

    if (not os.path.exists(new_file)) and os.path.exists(old_file):
        try:
            os.replace(old_file, new_file)
        except Exception:
            pass

    if os.path.exists(new_file):
        try:
            with open(new_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_custom_products(products):
    """保存自定义产品配置到文件"""
    config_dir = os.path.join(app_dir, "custom_products")
    try:
        os.makedirs(config_dir, exist_ok=True)
    except Exception:
        pass
    config_file = os.path.join(config_dir, "custom_products.json")
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(products, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False

def _migrate_legacy_files_once() -> None:
    if st.session_state.get('_legacy_files_migrated', False):
        return
    st.session_state['_legacy_files_migrated'] = True

    try:
        state_dir = os.path.join(app_dir, 'app_state')
        os.makedirs(state_dir, exist_ok=True)
        for name in os.listdir(app_dir):
            if name.startswith('app_state_') and name.endswith('.json'):
                old_path = os.path.join(app_dir, name)
                new_path = os.path.join(state_dir, name)
                if os.path.exists(old_path) and (not os.path.exists(new_path)):
                    try:
                        os.replace(old_path, new_path)
                    except Exception:
                        pass
    except Exception:
        pass

    try:
        old_custom = os.path.join(app_dir, 'custom_products.json')
        new_custom_dir = os.path.join(app_dir, 'custom_products')
        new_custom = os.path.join(new_custom_dir, 'custom_products.json')
        os.makedirs(new_custom_dir, exist_ok=True)
        if os.path.exists(old_custom) and (not os.path.exists(new_custom)):
            try:
                os.replace(old_custom, new_custom)
            except Exception:
                pass
    except Exception:
        pass

def get_session_id() -> str:
    """获取当前会话的唯一ID（浏览器会话级别）。"""
    if 'session_id' not in st.session_state:
        # 使用 UUID4 生成随机会话ID
        st.session_state['session_id'] = uuid.uuid4().hex
    return st.session_state['session_id']


def _get_state_file_path() -> str:
    """根据会话ID生成独立的状态文件路径。"""
    session_id = get_session_id()
    # 为不同会话分别保存，如 app_state_<session_id>.json
    state_dir = os.path.join(app_dir, "app_state")
    try:
        os.makedirs(state_dir, exist_ok=True)
    except Exception:
        pass
    return os.path.join(state_dir, f"app_state_{session_id}.json")

def _maybe_migrate_state_file(state_file: str) -> str:
    if not state_file:
        return state_file
    try:
        if os.path.exists(state_file):
            return state_file
        base = os.path.basename(state_file)
        old_file = os.path.join(app_dir, base)
        if os.path.exists(old_file):
            try:
                os.makedirs(os.path.dirname(state_file), exist_ok=True)
            except Exception:
                pass
            try:
                os.replace(old_file, state_file)
            except Exception:
                pass
    except Exception:
        pass
    return state_file


def load_app_state():
    """从会话专属文件加载应用状态（订单配置、模型路径、仿真结果等）。"""
    _migrate_legacy_files_once()
    state_file = _maybe_migrate_state_file(_get_state_file_path())
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def get_language():
    """获取当前语言设置"""
    if 'language' not in st.session_state:
        # 尝试从保存的状态中加载
        saved_state = load_app_state()
        st.session_state['language'] = saved_state.get('language', 'zh-CN')
    return st.session_state['language']

def clear_simulation_results():
    """清空调度结果（订单配置改变后，旧结果失效）"""
    st.session_state['show_results'] = False
    if 'final_stats' in st.session_state:
        del st.session_state['final_stats']
    if 'gantt_history' in st.session_state:
        del st.session_state['gantt_history']
    if 'score' in st.session_state:
        del st.session_state['score']
    if 'total_reward' in st.session_state:
        del st.session_state['total_reward']
    if 'heuristic_results' in st.session_state:
        del st.session_state['heuristic_results']

def save_app_state():
    """保存应用状态到当前会话专属文件"""
    _migrate_legacy_files_once()
    state_file = _maybe_migrate_state_file(_get_state_file_path())
    try:
        # 准备要保存的状态
        state_to_save = {
            'orders': st.session_state.get('orders', []),
            'model_path': st.session_state.get('model_path', ''),
            'model_loaded': st.session_state.get('model_loaded', False),
            'language': st.session_state.get('language', 'zh-CN'),
            'last_simulation': {
                'stats': st.session_state.get('last_stats', None),
                'gantt_history': st.session_state.get('last_gantt_history', None),
                'score': st.session_state.get('last_score', None),
                'total_reward': st.session_state.get('last_total_reward', None),
                'heuristic_results': st.session_state.get('heuristic_results', None)
            },
            # 订单配置界面 UI 参数（用于刷新后还原随机订单生成区的设置）
            'ui_config': {
                'config_method': st.session_state.get('config_method_key', 'random'),
                'num_orders': st.session_state.get('num_orders', 5),
                'min_quantity': st.session_state.get('qty_min', 3),
                'max_quantity': st.session_state.get('qty_max', 10),
                'min_due': st.session_state.get('due_min', 200),
                'max_due': st.session_state.get('due_max', 700),
                'min_arrival': st.session_state.get('arrival_min', 0),
                'max_arrival': st.session_state.get('arrival_max', 50),
                'max_steps_single': st.session_state.get('max_steps_single', 1500),
                # 动态环境配置开关
                'enable_failure': st.session_state.get('enable_failure', False),
                'enable_emergency': st.session_state.get('enable_emergency', False),
                # 是否启用启发式算法对比
                'compare_heuristics': st.session_state.get('compare_heuristics', True),
            },
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state_to_save, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        # 静默失败，不显示错误
        return False

def _extract_model_tag_from_path(model_path: str) -> str:
    import re
    if not model_path:
        return "unknown"
    base_name = os.path.basename(model_path)
    name, _ = os.path.splitext(base_name)
    match = re.match(r"^(\d{4}_\d{4})", name)
    if match:
        return match.group(1)
    parent = os.path.basename(os.path.dirname(model_path))
    match = re.match(r"^(\d{4}_\d{4})", parent)
    if match:
        return match.group(1)
    return name

def save_schedule_result(model_path: str, orders: list, final_stats: dict, score: float, gantt_history: list = None, heuristic_results: dict = None, enable_failure: bool = False, enable_emergency: bool = False, seeds_used: list = None) -> None:
    try:
        total_parts = sum(order.get("quantity", 0) for order in (orders or []))
        model_tag = _extract_model_tag_from_path(model_path)
        folder_name = f"{model_tag}+{total_parts}+{score:.3f}"
        
        # 创建独立的子文件夹
        save_dir = os.path.join(app_dir, "result", folder_name)
        os.makedirs(save_dir, exist_ok=True)

        # 处理启发式对比数据
        comparison_data = {}
        if heuristic_results:
            for name, res in heuristic_results.items():
                h_stats = res.get('stats', {})
                h_score = res.get('score', 0)
                comparison_data[name] = {
                    "score": h_score,
                    "makespan": h_stats.get('makespan', 0),
                    "total_parts": h_stats.get('total_parts', 0),
                    "mean_utilization": h_stats.get('mean_utilization', 0),
                    "total_tardiness": h_stats.get('total_tardiness', 0)
                }
                
                # 保存启发式算法的甘特图
                h_history = res.get('history', [])
                if h_history:
                    try:
                        h_fig = create_gantt_chart(h_history)
                        if h_fig:
                            h_fig.write_html(os.path.join(save_dir, f"gantt_{name}.html"))
                    except Exception:
                        pass

        # 保存MARL的甘特图
        if gantt_history:
            try:
                marl_fig = create_gantt_chart(gantt_history)
                if marl_fig:
                    marl_fig.write_html(os.path.join(save_dir, "gantt_MARL.html"))
            except Exception:
                pass

        # 保存详细JSON数据
        # 12-01 优化：从JSON中移除冗余的甘特图数据（已保存为HTML）
        clean_stats = final_stats.copy() if final_stats else {}
        if 'gantt_history' in clean_stats:
            del clean_stats['gantt_history']
            
        score_summary = {
            "type": "single_simulation",
            "items": []
        }
        score_summary["items"].append({
            "name": "MARL",
            "runs": 1,
            "avg_score": float(score),
            "seeds_used": (seeds_used if seeds_used is not None else [])
        })
        if heuristic_results:
            for name, res in heuristic_results.items():
                try:
                    score_summary["items"].append({
                        "name": str(name),
                        "runs": 1,
                        "avg_score": float(res.get('score', 0.0)),
                        "seeds_used": (seeds_used if seeds_used is not None else [])
                    })
                except Exception:
                    continue

        payload = {
            "model_path": model_path,
            "model_tag": model_tag,
            "total_parts": total_parts,
            "score": float(score),
            "final_stats": clean_stats,
            "heuristic_comparison": comparison_data,
            # 新增：保存订单列表
            "orders": orders if orders else [],
            # 新增：保存动态环境配置（包含具体参数）
            "dynamic_environment": {
                "equipment_failure": {
                    "enabled": enable_failure,
                    "parameters": {
                        "mtbf_hours": 24,                  # 平均故障间隔时间（小时）
                        "mttr_minutes": 30,                # 平均修复时间（分钟）
                        "failure_probability": 0.02        # 每小时故障概率
                    } if enable_failure else None
                },
                "emergency_orders": {
                    "enabled": enable_emergency,
                    "parameters": {
                        "arrival_rate": 0.1,               # 每小时紧急订单到达率
                        "priority_boost": 0,               # 紧急订单优先级提升
                        "due_date_reduction": 0.7          # 交期缩短比例（0.7表示缩短30%）
                    } if enable_emergency else None
                }
            },
            "seeds_used": seeds_used if seeds_used is not None else [],
            "score_summary": score_summary,
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "simulation_limits": {
                "max_steps": int(st.session_state.get('max_steps_single', 1500)),
                "max_sim_time": float(st.session_state.get('max_steps_single', 1500)),
            },
        }

        with open(os.path.join(save_dir, "result.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            
    except Exception:
        pass

def save_model_comparison_results(comparison_results: dict, model_names: list, seeds_used: list = None, heuristic_baselines: dict = None) -> None:
    """保存模型对比结果到文件"""
    try:
        # 生成对比结果文件夹名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"comparison_{timestamp}_{len(model_names)}models"
        
        # 创建对比结果文件夹
        save_dir = os.path.join(app_dir, "result", folder_name)
        os.makedirs(save_dir, exist_ok=True)
        
        # 先保存甘特图HTML，并在JSON里仅保存文件名引用（不保存gantt_history本体）
        gantt_file_map: Dict[str, Dict[int, str]] = {}
        for model_name, runs in comparison_results.items():
            gantt_file_map[model_name] = {}
            for i, run_data in enumerate(runs):
                if run_data.get('gantt_history'):
                    try:
                        fig = create_gantt_chart(run_data['gantt_history'])
                        if fig:
                            filename = f"gantt_{model_name.replace('/', '_')}_run{i+1}.html"
                            fig.write_html(os.path.join(save_dir, filename))
                            gantt_file_map[model_name][i] = filename
                    except Exception:
                        pass

        heuristic_gantt_file_map: Dict[str, Dict[int, str]] = {}
        if heuristic_baselines:
            for heuristic_name, runs in heuristic_baselines.items():
                heuristic_gantt_file_map[heuristic_name] = {}
                for i, run_data in enumerate(runs or []):
                    if run_data.get('gantt_history'):
                        try:
                            fig = create_gantt_chart(run_data['gantt_history'])
                            if fig:
                                filename = f"gantt_{heuristic_name}_run{i+1}.html"
                                fig.write_html(os.path.join(save_dir, filename))
                                heuristic_gantt_file_map[heuristic_name][i] = filename
                        except Exception:
                            pass

        # 准备对比数据（清理gantt_history）
        comparison_data = {}
        for model_name, runs in comparison_results.items():
            # 计算平均值
            avg_stats = {}
            if runs:
                avg_stats = {
                    'avg_completion_rate': sum(r['stats']['total_parts'] for r in runs) / len(runs),
                    'avg_makespan': sum(r['stats']['makespan'] for r in runs) / len(runs),
                    'avg_utilization': sum(r['stats']['mean_utilization'] for r in runs) / len(runs),
                    'avg_tardiness': sum(r['stats']['total_tardiness'] for r in runs) / len(runs),
                    'avg_score': sum(r['score'] for r in runs) / len(runs),
                    'avg_reward': sum(r.get('total_reward', 0.0) for r in runs) / len(runs),
                    'runs_count': len(runs)
                }

            cleaned_runs = []
            for i, r in enumerate(runs or []):
                r2 = dict(r)
                if 'gantt_history' in r2:
                    del r2['gantt_history']
                if model_name in gantt_file_map and i in gantt_file_map[model_name]:
                    r2['gantt_html'] = gantt_file_map[model_name][i]
                cleaned_runs.append(r2)

            comparison_data[model_name] = {
                'average_stats': avg_stats,
                'detailed_runs': cleaned_runs
            }

        cleaned_heuristic_baselines = {}
        if heuristic_baselines:
            for heuristic_name, runs in heuristic_baselines.items():
                cleaned_runs = []
                for i, r in enumerate(runs or []):
                    r2 = dict(r)
                    if 'gantt_history' in r2:
                        del r2['gantt_history']
                    if heuristic_name in heuristic_gantt_file_map and i in heuristic_gantt_file_map[heuristic_name]:
                        r2['gantt_html'] = heuristic_gantt_file_map[heuristic_name][i]
                    cleaned_runs.append(r2)
                cleaned_heuristic_baselines[heuristic_name] = cleaned_runs
        
        score_summary_items = []
        for model_name, runs in (comparison_results or {}).items():
            if runs:
                try:
                    score_summary_items.append({
                        "name": str(model_name),
                        "runs": len(runs),
                        "avg_score": float(sum(r.get('score', 0.0) for r in runs) / len(runs)),
                        "seeds_used": [int(r.get('seed')) for r in runs if r.get('seed') is not None]
                    })
                except Exception:
                    pass
        for h_name, runs in (heuristic_baselines or {}).items():
            if runs:
                try:
                    score_summary_items.append({
                        "name": str(h_name),
                        "runs": len(runs),
                        "avg_score": float(sum(r.get('score', 0.0) for r in runs) / len(runs)),
                        "seeds_used": [int(r.get('seed')) for r in runs if r.get('seed') is not None]
                    })
                except Exception:
                    pass

        # 构建保存的数据结构
        payload = {
            "comparison_type": "multi_model_performance",
            "models_compared": model_names,
            "seeds_used": seeds_used if seeds_used is not None else [],
            "comparison_results": comparison_data,
            "heuristic_baselines": cleaned_heuristic_baselines,
            # 保存当前订单配置
            "orders": st.session_state.get('orders', []),
            # 保存动态环境配置（包含具体参数）
            "dynamic_environment": {
                "equipment_failure": {
                    "enabled": st.session_state.get('enable_failure', False),
                    "parameters": {
                        "mtbf_hours": 24,                  # 平均故障间隔时间（小时）
                        "mttr_minutes": 30,                # 平均修复时间（分钟）
                        "failure_probability": 0.02        # 每小时故障概率
                    } if st.session_state.get('enable_failure', False) else None
                },
                "emergency_orders": {
                    "enabled": st.session_state.get('enable_emergency', False),
                    "parameters": {
                        "arrival_rate": 0.1,               # 每小时紧急订单到达率
                        "priority_boost": 0,               # 紧急订单优先级提升
                        "due_date_reduction": 0.7          # 交期缩短比例（0.7表示缩短30%）
                    } if st.session_state.get('enable_emergency', False) else None
                }
            },
            # 保存订单生成参数（如果是随机生成的）
            "order_generation_config": {
                "config_method": st.session_state.get('config_method_key', 'random'),
                "num_orders": st.session_state.get('num_orders', 5),
                "min_quantity": st.session_state.get('qty_min', 3),
                "max_quantity": st.session_state.get('qty_max', 10),
                "min_due": st.session_state.get('due_min', 200),
                "max_due": st.session_state.get('due_max', 700),
                "min_arrival": st.session_state.get('arrival_min', 0),
                "max_arrival": st.session_state.get('arrival_max', 50),
            },
            "comparison_parameters": {
                "max_steps": st.session_state.get('max_steps_single', 1500),
                "max_sim_time": st.session_state.get('max_steps_single', 1500),
                "runs_per_model": st.session_state.get('comparison_runs', 1)
            },
            "score_summary": {
                "type": "multi_model_comparison",
                "items": score_summary_items
            },
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        # 保存JSON文件
        with open(os.path.join(save_dir, "comparison_result.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        
        # 甘特图HTML已在上方保存，这里无需重复保存
        
    except Exception as e:
        # 静默失败，不影响主流程
        pass

def calculate_product_total_time(product: str, product_routes: dict) -> float:
    """计算产品总加工时间"""
    route = product_routes.get(product, [])
    return sum(step["time"] for step in route)

def save_dynamic_event_ablation_results(result: dict) -> None:
    """保存动态事件消融测试结果到文件（JSON不包含gantt_history，甘特图另存HTML）"""
    try:
        from environments import w_factory_config
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = result.get('model_name', 'model')
        folder_name = f"ablation_{timestamp}_{model_name.replace('/', '_').replace(' ', '_')}"
        save_dir = os.path.join(app_dir, "result", folder_name)
        os.makedirs(save_dir, exist_ok=True)

        def _clean_runs(runs: list, prefix: str):
            cleaned = []
            for i, r in enumerate(runs or []):
                r2 = dict(r)
                if r2.get('gantt_history'):
                    try:
                        _etl = (r2.get('stats', {}) or {}).get('event_timeline')
                        fig = create_gantt_chart(r2['gantt_history'], _etl)
                        if fig:
                            filename = f"gantt_{prefix}_run{i+1}.html"
                            fig.write_html(os.path.join(save_dir, filename))
                            r2['gantt_html'] = filename
                    except Exception:
                        pass
                    try:
                        del r2['gantt_history']
                    except Exception:
                        pass
                cleaned.append(r2)
            return cleaned

        disabled_clean = _clean_runs(result.get('disabled', []), "MARL_OFF")
        enabled_clean = _clean_runs(result.get('enabled', []), "MARL_ON")

        heur_dis = {}
        heur_en = {}
        for h, runs in (result.get('heuristic_disabled', {}) or {}).items():
            heur_dis[h] = _clean_runs(runs, f"{h}_OFF")
        for h, runs in (result.get('heuristic_enabled', {}) or {}).items():
            heur_en[h] = _clean_runs(runs, f"{h}_ON")

        failure_params = st.session_state.get('failure_config')
        emergency_params = st.session_state.get('emergency_config')
        if st.session_state.get('enable_failure', False) and not failure_params:
            failure_params = dict(w_factory_config.EQUIPMENT_FAILURE)
        if st.session_state.get('enable_emergency', False) and not emergency_params:
            emergency_params = dict(w_factory_config.EMERGENCY_ORDERS)

        score_summary_items = []
        for group_name, group_data in {"OFF": {"MARL": disabled_clean, "heur": heur_dis}, "ON": {"MARL": enabled_clean, "heur": heur_en}}.items():
            runs = (group_data.get("MARL") or [])
            if runs:
                try:
                    score_summary_items.append({
                        "group": group_name,
                        "name": "MARL",
                        "runs": len(runs),
                        "avg_score": float(sum(r.get('score', 0.0) for r in runs) / len(runs)),
                        "seeds_used": [int(r.get('seed')) for r in runs if r.get('seed') is not None]
                    })
                except Exception:
                    pass
            for h_name, hruns in (group_data.get("heur") or {}).items():
                if hruns:
                    try:
                        score_summary_items.append({
                            "group": group_name,
                            "name": str(h_name),
                            "runs": len(hruns),
                            "avg_score": float(sum(r.get('score', 0.0) for r in hruns) / len(hruns)),
                            "seeds_used": [int(r.get('seed')) for r in hruns if r.get('seed') is not None]
                        })
                    except Exception:
                        pass

        payload = {
            "type": "dynamic_event_ablation",
            "model_name": result.get('model_name'),
            "model_path": result.get('model_path'),
            "seeds_used": result.get('seeds_used', []),
            "orders": st.session_state.get('orders', []),
            "dynamic_environment_on": {
                "equipment_failure": {
                    "enabled": st.session_state.get('enable_failure', False),
                    "parameters": failure_params if st.session_state.get('enable_failure', False) else None
                },
                "emergency_orders": {
                    "enabled": st.session_state.get('enable_emergency', False),
                    "parameters": emergency_params if st.session_state.get('enable_emergency', False) else None
                }
            },
            "runs": {
                "OFF": {
                    "MARL": disabled_clean,
                    "heuristics": heur_dis,
                },
                "ON": {
                    "MARL": enabled_clean,
                    "heuristics": heur_en,
                }
            },
            "score_summary": {
                "type": "dynamic_event_ablation",
                "items": score_summary_items
            },
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(os.path.join(save_dir, "ablation_result.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def validate_order_config(orders: list, custom_products: dict = None) -> dict:
    """
    验证订单配置的合理性，并预测运行结果
    
    返回格式：
    {
        'valid': bool,
        'warnings': list,
        'info': dict,
        'difficulty_level': str
    }
    """
    lang = get_language()  # 获取当前语言

    # 合并系统产品和自定义产品
    all_product_routes = PRODUCT_ROUTES.copy()
    if custom_products:
        all_product_routes.update(custom_products)
    
    warnings = []
    info = {}
    
    # 1. 检查订单中的产品是否都有工艺路线
    order_products = set(order["product"] for order in orders)
    defined_products = set(all_product_routes.keys())
    
    if not order_products.issubset(defined_products):
        missing = order_products - defined_products
        return {
            'valid': False,
            'warnings': [get_text("error_missing_route", lang, ', '.join(missing))],
            'info': {},
            'difficulty_level': 'invalid'
        }
    
    # 2. 计算基础统计
    total_parts = sum(order["quantity"] for order in orders)
    total_processing_time = 0
    
    for order in orders:
        product_time = calculate_product_total_time(order["product"], all_product_routes)
        total_processing_time += product_time * order["quantity"]
    
    info['total_parts'] = total_parts
    info['total_processing_time'] = total_processing_time
    
    # 3. 计算瓶颈工作站的理论最小完工时间
    bottleneck_time = {}
    for station_name, station_config in WORKSTATIONS.items():
        station_load = 0
        for order in orders:
            route = all_product_routes.get(order["product"], [])
            for step in route:
                if step["station"] == station_name:
                    station_load += step["time"] * order["quantity"]
        
        # 考虑设备数量的并行处理能力
        bottleneck_time[station_name] = station_load / station_config["count"]
    
    theoretical_makespan = max(bottleneck_time.values()) if bottleneck_time else 0
    bottleneck_station = max(bottleneck_time, key=bottleneck_time.get) if bottleneck_time else "未知"
    
    info['theoretical_makespan'] = theoretical_makespan
    info['bottleneck_station'] = bottleneck_station
    info['bottleneck_load'] = bottleneck_time.get(bottleneck_station, 0)
    
    # 4. 检查交期合理性
    min_due_date = min(order["due_date"] for order in orders)
    max_due_date = max(order["due_date"] for order in orders)
    avg_due_date = np.mean([order["due_date"] for order in orders])
    
    info['min_due_date'] = min_due_date
    info['max_due_date'] = max_due_date
    info['avg_due_date'] = avg_due_date
    
    # 5. 检查订单到达时间
    if any('arrival_time' in order for order in orders):
        arrival_times = [order.get('arrival_time', 0) for order in orders]
        info['has_arrival_time'] = True
        info['max_arrival_time'] = max(arrival_times)
    else:
        info['has_arrival_time'] = False
    
    # 6. 评估难度等级和生成警告
    simulation_time = SIMULATION_TIME
    makespan_ratio = theoretical_makespan / simulation_time
    
    if makespan_ratio > 1.0:
        difficulty_level = get_text("difficulty_very_high", lang)
        warnings.append(get_text("warn_makespan_too_long", lang, theoretical_makespan, simulation_time))
        warnings.append(get_text("suggestion_reduce_orders", lang))
    elif makespan_ratio > 0.8:
        difficulty_level = get_text("difficulty_high", lang)
        warnings.append(get_text("info_high_challenge", lang, makespan_ratio * 100))
    elif makespan_ratio > 0.5:
        difficulty_level = get_text("difficulty_medium", lang)
        warnings.append(get_text("info_medium_challenge", lang, makespan_ratio * 100))
    else:
        difficulty_level = get_text("difficulty_low", lang)
        warnings.append(get_text("info_low_challenge", lang, makespan_ratio * 100))
    
    # 7. 检查交期是否合理
    if min_due_date < theoretical_makespan * 0.5:
        warnings.append(get_text("warn_due_date_too_short", lang, min_due_date))
    
    if theoretical_makespan > avg_due_date:
        warnings.append(get_text("warn_avg_due_date_too_short", lang, avg_due_date, theoretical_makespan))
    
    # 8. 检查瓶颈工作站
    bottleneck_ratio = info['bottleneck_load'] / simulation_time
    if bottleneck_ratio > 0.9:
        warnings.append(get_text("warn_bottleneck_overload", lang, bottleneck_station, bottleneck_ratio * 100))
    elif bottleneck_ratio > 0.7:
        warnings.append(get_text("warn_bottleneck_high_load", lang, bottleneck_station, bottleneck_ratio * 100))
    
    return {
        'valid': True,
        'warnings': warnings,
        'info': info,
        'difficulty_level': difficulty_level
    }

def extract_orders_from_json_obj(obj):
    if obj is None:
        return None
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        if isinstance(obj.get('orders'), list):
            return obj.get('orders')
        if isinstance(obj.get('order_config'), list):
            return obj.get('order_config')
        if isinstance(obj.get('order_list'), list):
            return obj.get('order_list')
        if isinstance(obj.get('config'), dict):
            nested = obj.get('config')
            if isinstance(nested.get('orders'), list):
                return nested.get('orders')
    return None

def normalize_orders_list(orders):
    if not isinstance(orders, list):
        return None
    normalized = []
    for o in orders:
        if not isinstance(o, dict):
            return None
        if 'product' not in o or 'quantity' not in o or 'priority' not in o or 'due_date' not in o:
            return None
        normalized.append({
            'product': o.get('product'),
            'quantity': int(o.get('quantity')),
            'priority': int(o.get('priority')),
            'arrival_time': int(o.get('arrival_time', 0)),
            'due_date': int(o.get('due_date')),
        })
    return normalized

def _load_model_meta_robust(model_path: str):
    """尽可能从磁盘找到与模型对应的 meta.json（兼容多种目录/命名结构）。"""
    if not model_path:
        return None

    def _try_load(path: str):
        if path and os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    # 规则1：按约定的 base_path_meta.json
    base_path = model_path.replace('.keras', '').replace('.h5', '').replace('_actor', '')
    meta = _try_load(f"{base_path}_meta.json")
    if meta is not None:
        return meta

    # 规则2：同目录下找与文件名前缀最匹配的 *_meta.json
    try:
        model_dir = os.path.dirname(model_path)
        model_file = os.path.basename(model_path)
        model_stem = model_file
        if model_stem.endswith('.keras'):
            model_stem = model_stem[:-6]
        if model_stem.endswith('.h5'):
            model_stem = model_stem[:-3]
        if model_stem.endswith('_actor'):
            model_stem = model_stem[:-6]

        candidates = [
            os.path.join(model_dir, f)
            for f in (os.listdir(model_dir) if os.path.isdir(model_dir) else [])
            if f.endswith('_meta.json')
        ]
        # 优先：同名
        for p in candidates:
            if os.path.basename(p) == f"{model_stem}_meta.json":
                meta = _try_load(p)
                if meta is not None:
                    return meta
        # 次优：以 model_stem 开头
        for p in candidates:
            if os.path.basename(p).startswith(model_stem):
                meta = _try_load(p)
                if meta is not None:
                    return meta
        # 兜底：目录里任意一个 meta
        for p in candidates:
            meta = _try_load(p)
            if meta is not None:
                return meta
    except Exception:
        pass

    return None

@st.cache_resource
def load_model(model_path):
    """加载训练好的模型，同时返回元数据"""
    try:
        # 10-26-16-00 使用健壮的加载函数
        actor_model = load_actor_model_robust(model_path)
        if actor_model is None:
            return None, None, get_text("error_load_model_failed", get_language(), "所有加载策略均失败")
        
        # 尝试加载元数据文件（健壮模式）
        meta_data = _load_model_meta_robust(model_path)

        # 兼容：历史字段名
        if isinstance(meta_data, dict):
            if ('environment_config' not in meta_data) and isinstance(meta_data.get('env_config'), dict):
                meta_data['environment_config'] = meta_data.get('env_config')
            if ('save_timestamp' not in meta_data) and meta_data.get('timestamp'):
                meta_data['save_timestamp'] = meta_data.get('timestamp')
        
        return actor_model, meta_data, get_text("model_loaded_successfully", get_language())
    except Exception as e:
        return None, None, get_text("error_load_model_failed", get_language(), str(e))

def find_available_models():
    """
    查找所有可用的训练模型。
    会搜索两种路径：
    1. 旧版路径: mappo/ppo_models/<timestamp>/model.keras
    2. 新版路径 (通过 auto_train.py 创建): <experiment_dir>/models/<timestamp>/model.keras
    """
    models = []
    
    # --- 搜索新版路径 ---
    # 遍历项目根目录下的所有条目
    for experiment_dir in os.listdir(project_root):
        exp_path = os.path.join(project_root, experiment_dir)
        # 必须是一个目录
        if not os.path.isdir(exp_path):
            continue
        
        # 检查目录下是否存在 'models' 子目录
        models_path = os.path.join(exp_path, "models")
        if os.path.exists(models_path) and os.path.isdir(models_path):
            # 如果存在，则认为这是一个实验目录
            for timestamp_dir in os.listdir(models_path):
                run_path = os.path.join(models_path, timestamp_dir)
                if os.path.isdir(run_path):
                    # 🔧 修复：支持嵌套的时间戳子目录结构
                    # 新结构：models/<timestamp1>/<timestamp2>/<model_file>
                    # 旧结构：models/<timestamp>/<model_file>
                    
                    # 首先尝试在当前目录直接查找模型文件（旧结构）
                    found_in_current = False
                    for file in os.listdir(run_path):
                        if file.endswith("_actor.keras"):
                            model_path = os.path.join(run_path, file)
                            model_name = file.replace("_actor.keras", "")
                            models.append({
                                "name": f"{experiment_dir}/{model_name}",
                                "path": model_path,
                            })
                            found_in_current = True
                    
                    # 如果当前目录没找到，递归查找子目录（新结构）
                    if not found_in_current:
                        for sub_item in os.listdir(run_path):
                            sub_path = os.path.join(run_path, sub_item)
                            if os.path.isdir(sub_path):
                                for file in os.listdir(sub_path):
                                    if file.endswith("_actor.keras"):
                                        model_path = os.path.join(sub_path, file)
                                        model_name = file.replace("_actor.keras", "")
                                        models.append({
                                            "name": f"{experiment_dir}/{model_name}",
                                            "path": model_path,
                                        })

    # --- 搜索旧版路径 (用于兼容) ---
    old_models_path = os.path.join(project_root, "mappo", "ppo_models")
    if os.path.exists(old_models_path):
        for timestamp_dir in os.listdir(old_models_path):
            dir_path = os.path.join(old_models_path, timestamp_dir)
            if os.path.isdir(dir_path):
                for file in os.listdir(dir_path):
                    if file.endswith("_actor.keras"):
                        model_path = os.path.join(dir_path, file)
                        model_name = file.replace("_actor.keras", "")
                        models.append({
                            # 为旧版模型添加 "legacy" 前缀以区分
                            "name": f"legacy/{timestamp_dir}/{model_name}",
                            "path": model_path,
                        })

    # 按路径对模型列表进行降序排序，确保最新的模型显示在最前面
    models.sort(key=lambda x: x['path'], reverse=True)
    
    return models

def run_heuristic_scheduling(heuristic_name, orders_config, custom_products=None, max_steps=1500, progress_bar=None, status_text=None, enable_failure=False, enable_emergency=False, failure_config=None, emergency_config=None, seed: int = 42, progress_callback=None, progress_callback_interval: int = 10):
    """
    运行启发式算法调度仿真
    
    🔧 直接复用 evaluation.py 中的启发式策略实现，确保两个脚本使用完全一致的评估逻辑
    """
    from environments import w_factory_config
    from environments.w_factory_env import calculate_slack_time
    import copy
    from environments.w_factory_config import EVALUATION_CONFIG
    
    original_routes = None
    if custom_products:
        original_routes = w_factory_config.PRODUCT_ROUTES.copy()
        w_factory_config.PRODUCT_ROUTES.update(custom_products)
    
    try:
        config = {
            'custom_orders': orders_config,
            'equipment_failure_enabled': enable_failure,
            'emergency_orders_enabled': enable_emergency,
            'stage_name': f'{heuristic_name}启发式调度',
            'MAX_SIM_STEPS': int(max_steps),
            'MAX_SIM_TIME': float(max_steps),
        }
        
        # 12-02 新增：合并设备故障和紧急插单的高级配置参数
        if enable_failure and failure_config:
            config.update({
                'equipment_failure_config': failure_config
            })
        
        if enable_emergency and emergency_config:
            config.update({
                'emergency_orders_config': emergency_config
            })
        
        if status_text:
            status_text.text(f"初始化{heuristic_name}算法...")
        
        # 🔧 与 evaluation.py 保持一致：合并基础配置
        final_config = copy.deepcopy(EVALUATION_CONFIG)
        final_config.update(config)
        
        env = WFactoryEnv(config=final_config)
        obs, info = env.reset(seed=seed)
        sim = env.sim
        
        step_count = 0
        
        if status_text:
            status_text.text(f"运行{heuristic_name}调度...")
        
        # 🌟 关键修改：直接复用 evaluation.py 中的启发式策略函数（417-550行的逻辑）
        def heuristic_policy(obs, env, info, step_count):
            """
            启发式策略函数 - 完全复用 evaluation.py 的实现（417-550行）
            
            智能适配版：自动适配任何动作空间结构
            设计理念：
            1. 优先检测动作空间中是否存在启发式动作（向后兼容旧版本）
            2. 如果不存在，独立计算启发式逻辑并映射到候选动作（适配新版本）
            3. 完全解耦启发式算法与动作空间设计
            """
            actions = {}
            
            # 获取动作名称映射
            action_names = []
            info_source = info
            
            if env.agents:
                first_agent = env.agents[0]
                if info_source and first_agent in info_source:
                    action_names = info_source[first_agent].get('obs_meta', {}).get('action_names', [])
            
            action_map = {name: idx for idx, name in enumerate(action_names)}
            
            # 定义启发式名称到动作名称的映射
            heuristic_to_action_map = {
                'FIFO': 'FIFO',
                'EDD': 'URGENT_EDD',
                'SPT': 'SHORT_SPT',
                'ATC': 'ATC',
            }
            
            target_action_name = heuristic_to_action_map.get(heuristic_name)
            use_direct_action = (target_action_name in action_map)
            
            for agent_id in env.agents:
                station_name = agent_id.replace("agent_", "")
                queue = sim.queues[station_name].items
                
                if not queue:
                    sp = env.action_space(agent_id)
                    if isinstance(sp, gym.spaces.MultiDiscrete):
                        actions[agent_id] = np.zeros(len(sp.nvec), dtype=sp.dtype)
                    else:
                        actions[agent_id] = 0
                    continue
                
                # 分支1：动作空间中存在启发式动作
                if use_direct_action:
                    sp = env.action_space(agent_id)
                    if isinstance(sp, gym.spaces.MultiDiscrete):
                        k = len(sp.nvec)
                        actions[agent_id] = np.array([action_map[target_action_name]] * k, dtype=sp.dtype)
                    else:
                        actions[agent_id] = action_map[target_action_name]
                    continue
                
                # 分支2：动作空间中不存在启发式动作（独立实现）
                selected_parts = []
                
                if heuristic_name == 'FIFO':
                    selected_parts = [queue[0]]
                elif heuristic_name == 'EDD':
                    selected_parts = sorted(queue, key=lambda p: calculate_slack_time(p, sim.env.now, sim.queues))
                elif heuristic_name == 'SPT':
                    selected_parts = sorted(queue, key=lambda p: p.get_processing_time())
                elif heuristic_name == 'ATC':
                    now = float(sim.env.now)
                    procs = [max(1e-6, float(p.get_processing_time())) for p in queue]
                    p_bar = float(np.mean(procs)) if procs else 1.0
                    k = 2.0
                    scored = []
                    for p in queue:
                        proc = max(1e-6, float(p.get_processing_time()))
                        slack = float(calculate_slack_time(p, sim.env.now, sim.queues))
                        slack_pos = max(0.0, slack)
                        score = float(np.exp(-slack_pos / max(1e-6, k * p_bar))) / proc
                        scored.append((p, score))
                    scored.sort(key=lambda x: x[1], reverse=True)
                    selected_parts = [p for p, _ in scored]
                else:
                    raise ValueError(f"未知的启发式规则: {heuristic_name}")
                
                # 映射到候选动作
                candidates = sim._get_candidate_workpieces(station_name)
                sp = env.action_space(agent_id)
                
                if isinstance(sp, gym.spaces.MultiDiscrete):
                    k = len(sp.nvec)
                    chosen_actions = []
                    used_part_ids = set()
                    
                    for target_part in selected_parts:
                        if len(chosen_actions) >= k:
                            break
                        if target_part.part_id in used_part_ids:
                            continue
                        
                        found = 0
                        for idx, cand in enumerate(candidates):
                            cand_part = cand.get("part") if isinstance(cand, dict) else cand[0]
                            if cand_part and cand_part.part_id == target_part.part_id:
                                candidate_action_start = next(
                                    (i for i, name in enumerate(action_names) if "CANDIDATE_" in name),
                                    1
                                )
                                found = candidate_action_start + idx
                                break
                        
                        if found != 0:
                            chosen_actions.append(int(found))
                            used_part_ids.add(target_part.part_id)
                    
                    while len(chosen_actions) < k:
                        chosen_actions.append(0)
                    
                    actions[agent_id] = np.array(chosen_actions, dtype=sp.dtype)
                else:
                    action = 0
                    if selected_parts:
                        target_part = selected_parts[0]
                        for idx, cand in enumerate(candidates):
                            cand_part = cand.get("part") if isinstance(cand, dict) else cand[0]
                            if cand_part and cand_part.part_id == target_part.part_id:
                                candidate_action_start = next(
                                    (i for i, name in enumerate(action_names) if "CANDIDATE_" in name),
                                    1
                                )
                                action = candidate_action_start + idx
                                break
                    actions[agent_id] = action
            
            return actions
        
        # 运行仿真循环
        while step_count < max_steps:
            actions = heuristic_policy(obs, env, info, step_count)
            obs, rewards, terminations, truncations, info = env.step(actions)
            step_count += 1
            
            if progress_bar and step_count % 10 == 0:
                progress = min(step_count / max_steps, 1.0)
                progress_bar.progress(progress)
                if status_text:
                    status_text.text(f"运行{heuristic_name}调度... ({step_count}/{max_steps})")
            
            if progress_callback and (step_count % int(progress_callback_interval) == 0):
                try:
                    progress_callback(int(step_count), int(max_steps))
                except Exception:
                    pass
            
            if any(terminations.values()) or any(truncations.values()):
                break

        if progress_callback:
            try:
                progress_callback(int(max_steps), int(max_steps))
            except Exception:
                pass
        
        if status_text:
            status_text.text(f"{heuristic_name}调度完成")
        
        if progress_bar:
            progress_bar.progress(1.0)
        
        final_stats = env.sim.get_final_stats()
        gantt_history = env.sim.gantt_chart_history
        score = calculate_episode_score(final_stats, final_config)

        try:
            last_info = info or {}
            first_agent = env.agents[0] if getattr(env, 'agents', None) else None
            ainfo = last_info.get(first_agent, {}) if first_agent is not None else {}
            if isinstance(ainfo, dict):
                if 'terminal_score_bonus' in ainfo:
                    bonus_per_agent = float(ainfo.get('terminal_score_bonus', 0.0))
                    final_stats['terminal_score_bonus_per_agent'] = bonus_per_agent
                    final_stats['terminal_score_bonus_total'] = float(bonus_per_agent) * float(max(1, len(getattr(env, 'agents', []) or [])))
                if 'episode_score_baseline' in ainfo:
                    final_stats['episode_score_baseline'] = float(ainfo.get('episode_score_baseline', 0.0))
                if 'episode_score_delta' in ainfo:
                    final_stats['episode_score_delta'] = float(ainfo.get('episode_score_delta', 0.0))
                if 'episode_score' in ainfo:
                    final_stats['episode_score_from_env'] = float(ainfo.get('episode_score', 0.0))
        except Exception:
            pass
        
        env.close()
        
        return final_stats, gantt_history, score
    
    finally:
        if original_routes is not None:
            w_factory_config.PRODUCT_ROUTES = original_routes

def run_scheduling(actor_model, orders_config, custom_products=None, max_steps=1500, progress_bar=None, status_text=None, enable_failure=False, enable_emergency=False, failure_config=None, emergency_config=None, seed: int = 42, progress_callback=None, progress_callback_interval: int = 10):
    """运行调度仿真"""
    # 如果有自定义产品，临时添加到PRODUCT_ROUTES
    from environments import w_factory_config
    original_routes = None
    
    if custom_products:
        original_routes = w_factory_config.PRODUCT_ROUTES.copy()
        w_factory_config.PRODUCT_ROUTES.update(custom_products)
    
    try:
        # 10-27-16-30 统一：环境端兼容 'disable_failures'，但为避免误解，这里明确使用 equipment_failure_enabled
        config = {
            'custom_orders': orders_config,
            'equipment_failure_enabled': enable_failure,
            'emergency_orders_enabled': enable_emergency,
            'stage_name': '用户自定义调度',
            'MAX_SIM_STEPS': int(max_steps),
            'MAX_SIM_TIME': float(max_steps),
        }
        
        # 12-02 新增：合并设备故障和紧急插单的高级配置参数
        if enable_failure and failure_config:
            config.update({
                'equipment_failure_config': failure_config
            })
        
        if enable_emergency and emergency_config:
            config.update({
                'emergency_orders_config': emergency_config
            })
        
        if status_text:
            status_text.text(get_text("initializing", get_language()))
        
        env = WFactoryEnv(config=config)
        obs, info = env.reset(seed=seed)
        
        step_count = 0
        total_reward = 0
        
        if status_text:
            status_text.text(get_text("starting_sim", get_language()))
        
        while step_count < max_steps:
            actions = {}
            for agent in env.agents:
                if agent in obs:
                    # 兼容旧模型：若环境观测维度与模型输入不一致，则自动截断/补零
                    try:
                        expected_dim = int(getattr(actor_model, 'input_shape', [None, None])[-1])
                    except Exception:
                        expected_dim = None

                    x = np.asarray(obs[agent], dtype=np.float32).reshape(-1)
                    if expected_dim is not None and expected_dim > 0 and x.shape[0] != expected_dim:
                        if x.shape[0] > expected_dim:
                            x = x[:expected_dim]
                        else:
                            pad = expected_dim - x.shape[0]
                            x = np.concatenate([x, np.zeros((pad,), dtype=np.float32)], axis=0)

                    state = tf.expand_dims(x, 0)
                    # 10-25-14-30 统一：兼容多头/单头输出并采用按头无放回贪心选择
                    action_probs_tensor = actor_model(state, training=False)
                    if isinstance(action_probs_tensor, (list, tuple)):
                        head_probs_list = [np.squeeze(h.numpy()) for h in action_probs_tensor]
                    else:
                        head_probs_list = [np.squeeze(action_probs_tensor.numpy()[0])]
                    
                    # 11-26 修复：应用动作掩码 (与 evaluation.py 保持一致)
                    # 解决离线评估0完成问题：若无掩码，策略容易选到被收紧的IDLE
                    if info and agent in info and 'action_mask' in info[agent]:
                        action_mask = info[agent]['action_mask']
                        masked_head_probs_list = []
                        for probs in head_probs_list:
                            # 复制一份以避免修改原数组
                            current_probs = probs.copy()
                            
                            # 确保掩码长度匹配
                            if len(action_mask) == len(current_probs):
                                masked_probs = current_probs * action_mask
                                # 检查是否全零（所有动作都被掩码屏蔽）
                                if np.sum(masked_probs) > 1e-10:
                                    current_probs = masked_probs
                                # else: 若全被屏蔽，回退到原始概率（避免全零导致的错误）
                            
                            masked_head_probs_list.append(current_probs)
                        head_probs_list = masked_head_probs_list

                    sp = env.action_space(agent)
                    if isinstance(sp, gym.spaces.MultiDiscrete):
                        k = len(sp.nvec)
                        chosen = choose_parallel_actions_multihead(head_probs_list, k, greedy=True)
                        actions[agent] = np.array(chosen, dtype=sp.dtype)
                    else:
                        p = np.asarray(head_probs_list[0], dtype=np.float64)
                        p = np.clip(p, 1e-12, np.inf)
                        actions[agent] = int(np.argmax(p))
            
            obs, rewards, terminations, truncations, info = env.step(actions)
            total_reward += sum(rewards.values())
            step_count += 1
            
            # 更新进度条
            if progress_bar and step_count % 10 == 0:
                progress = min(step_count / max_steps, 1.0)
                progress_bar.progress(progress)
                if status_text:
                    status_text.text(get_text("scheduling", get_language(), step_count, max_steps))

            if progress_callback and (step_count % int(progress_callback_interval) == 0):
                try:
                    progress_callback(int(step_count), int(max_steps))
                except Exception:
                    pass
            
            if any(terminations.values()) or any(truncations.values()):
                break
        
        if status_text:
            status_text.text(get_text("generating_results", get_language()))
        
        if progress_bar:
            progress_bar.progress(1.0)
        
        final_stats = env.sim.get_final_stats()
        gantt_history = env.sim.gantt_chart_history
        score = calculate_episode_score(final_stats, config)

        try:
            last_info = info or {}
            first_agent = env.agents[0] if getattr(env, 'agents', None) else None
            ainfo = last_info.get(first_agent, {}) if first_agent is not None else {}
            if isinstance(ainfo, dict):
                if 'terminal_score_bonus' in ainfo:
                    bonus_per_agent = float(ainfo.get('terminal_score_bonus', 0.0))
                    final_stats['terminal_score_bonus_per_agent'] = bonus_per_agent
                    final_stats['terminal_score_bonus_total'] = float(bonus_per_agent) * float(max(1, len(getattr(env, 'agents', []) or [])))
                if 'episode_score_baseline' in ainfo:
                    final_stats['episode_score_baseline'] = float(ainfo.get('episode_score_baseline', 0.0))
                if 'episode_score_delta' in ainfo:
                    final_stats['episode_score_delta'] = float(ainfo.get('episode_score_delta', 0.0))
                if 'episode_score' in ainfo:
                    final_stats['episode_score_from_env'] = float(ainfo.get('episode_score', 0.0))
        except Exception:
            pass
        
        env.close()
        
        if status_text:
            status_text.text(get_text("scheduling_complete", get_language()))
        
        return final_stats, gantt_history, score, total_reward
    finally:
        # 恢复原始产品路线
        if original_routes is not None:
            w_factory_config.PRODUCT_ROUTES = original_routes

def create_gantt_chart(history, event_timeline=None):
    """创建交互式甘特图"""
    if not history:
        return None
    
    df = pd.DataFrame(history)
    
    fig = go.Figure()
    
    # 获取所有唯一的产品类型并分配颜色
    products = df['Product'].unique()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    color_map = {product: colors[i % len(colors)] for i, product in enumerate(products)}
    
    lang = get_language()
    
    # 为每个加工任务添加甘特图条
    for _, row in df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['Start'], row['Finish'], row['Finish'], row['Start'], row['Start']],
            y=[get_text(row['Resource'], lang), get_text(row['Resource'], lang), get_text(row['Resource'], lang), get_text(row['Resource'], lang), get_text(row['Resource'], lang)],
            fill='toself',
            fillcolor=color_map[row['Product']],
            line=dict(color=color_map[row['Product']], width=2),
            hovertemplate=f"<b>{row['Task']}</b><br>" +
                         f"{get_text('workstation', lang)}: {get_text(row['Resource'], lang)}<br>" +
                         f"{get_text('product', lang)}: {row['Product']}<br>" +
                         f"{get_text('part_id', lang)}: {row['Part ID']}<br>" +
                         f"{get_text('order_id', lang)}: {row['Order ID']}<br>" +
                         f"{get_text('start_time', lang)}: {row['Start']:.1f}{get_text('minutes', lang)}<br>" +
                         f"{get_text('end_time', lang)}: {row['Finish']:.1f}{get_text('minutes', lang)}<br>" +
                         f"{get_text('duration', lang)}: {row['Duration']:.1f}{get_text('minutes', lang)}<extra></extra>",
            name=row['Product'],
            showlegend=row['Product'] not in [trace.name for trace in fig.data]
        ))

    # 叠加动态事件时间线标注（故障区间 / 插单时刻）
    try:
        if event_timeline:
            failures = [e for e in (event_timeline or []) if isinstance(e, dict) and e.get('type') == 'failure']
            emerg = [e for e in (event_timeline or []) if isinstance(e, dict) and e.get('type') == 'emergency_order']

            # 故障：半透明区间（覆盖整个y轴）
            for i, e in enumerate(failures):
                try:
                    x0 = float(e.get('start', 0.0))
                    x1 = float(e.get('end', x0))
                    station = str(e.get('station', ''))
                    fig.add_shape(
                        type='rect',
                        xref='x',
                        yref='paper',
                        x0=x0,
                        x1=x1,
                        y0=0.0,
                        y1=1.0,
                        fillcolor='rgba(220, 53, 69, 0.12)',
                        line=dict(width=0),
                        layer='below'
                    )
                    fig.add_annotation(
                        x=(x0 + x1) / 2.0,
                        y=1.02,
                        xref='x',
                        yref='paper',
                        text=f"故障: {station}",
                        showarrow=False,
                        font=dict(size=10, color='rgba(220, 53, 69, 0.95)')
                    )
                except Exception:
                    pass

            # 插单：竖线 + 注释
            for i, e in enumerate(emerg):
                try:
                    t = float(e.get('time', 0.0))
                    oid = e.get('order_id', None)
                    qty = e.get('quantity', None)
                    fig.add_shape(
                        type='line',
                        xref='x',
                        yref='paper',
                        x0=t,
                        x1=t,
                        y0=0.0,
                        y1=1.0,
                        line=dict(color='rgba(255, 193, 7, 0.85)', width=2, dash='dot'),
                        layer='above'
                    )
                    label = "插单"
                    if oid is not None:
                        label += f"#{oid}"
                    if qty is not None:
                        label += f" x{qty}"
                    fig.add_annotation(
                        x=t,
                        y=1.08,
                        xref='x',
                        yref='paper',
                        text=label,
                        showarrow=False,
                        font=dict(size=10, color='rgba(255, 193, 7, 0.95)')
                    )
                except Exception:
                    pass
    except Exception:
        pass
    
    fig.update_layout(
        title=get_text('gantt_chart_title', lang),
        xaxis=dict(title=get_text('gantt_xaxis_title', lang), type='linear'),
        yaxis=dict(title=get_text('gantt_yaxis_title', lang), categoryorder="category ascending"),
        font=dict(family="Arial, sans-serif", size=12),
        hovermode='closest',
        height=500,
        showlegend=True
    )
    
    return fig

def create_utilization_chart(stats):
    """创建设备利用率柱状图"""
    utilization_data = stats.get('equipment_utilization', {})
    
    if not utilization_data:
        return None
    
    lang = get_language()
    
    df = pd.DataFrame([
        {get_text("workstation", lang): get_text(station, lang), get_text("utilization_rate", lang): util * 100}
        for station, util in utilization_data.items()
    ])
    
    fig = go.Figure(data=[
        go.Bar(
            x=df[get_text("workstation", lang)],
            y=df[get_text("utilization_rate", lang)],
            text=df[get_text("utilization_rate", lang)].apply(lambda x: f"{x:.1f}%"),
            textposition='auto',
            marker_color='steelblue'
        )
    ])
    
    fig.update_layout(
        title=get_text("util_chart_title", lang),
        xaxis=dict(title=get_text("workstation", lang)),
        yaxis=dict(title=get_text("utilization_rate_percent", lang)),
        height=400
    )
    
    return fig

# ============================================================================
# 主应用界面
# ============================================================================

def main():
    # 获取当前语言
    lang = get_language()
    
    # 添加自定义CSS样式，美化图标按钮
    st.markdown("""
        <style>
        /*
         * 1. 强制垂直对齐右上角的图标和选择器
         *    - 使用 :has() 选择器精确定位容器
         *    - align-items: center; 是对齐的关键
         */
        div[data-testid="stHorizontalBlock"]:has(div[data-testid="stSelectbox"]):has(button) {
            align-items: center;
        }

        /*
         * 2. 将次要按钮彻底变成无边框的图标按钮
         *    - 移除边框、背景和阴影
         *    - 鼠标悬停时提供一个微妙的背景反馈
         */
        button[data-testid="baseButton-secondary"] {
            border: none !important;
            background-color: transparent !important;
            box-shadow: none !important;
        }
        button[data-testid="baseButton-secondary"]:hover {
            background-color: rgba(0, 0, 0, 0.05) !important;
            border-radius: 0.5rem !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # 🎨 顶部布局：标题 + 右上角图标按钮
    col_title, col_spacer, col_icons = st.columns([5, 1, 2])
    
    with col_title:
        st.title(get_text("app_title", lang))
        st.markdown(f"**{get_text('app_subtitle', lang)}**")
    
    with col_icons:
        # 右上角两个图标按钮并排
        icon_col1, icon_col2 = st.columns([1, 1])
        
        with icon_col1:
            # 🌐 语言选择器（下拉菜单）
            current_lang = get_language()
            lang_options = list(LANGUAGES.keys())
            
            try:
                current_index = lang_options.index(current_lang)
            except:
                current_index = 0
            
            # 创建带图标的选项
            selected_lang = st.selectbox(
                "语言",
                options=lang_options,
                format_func=lambda x: f"🌐 {LANGUAGES[x]}",
                index=current_index,
                key="lang_selector",
                label_visibility="collapsed"
            )
            
            # 如果语言改变，更新并保存
            if selected_lang != current_lang:
                st.session_state['language'] = selected_lang
                save_app_state()
                st.rerun()
        
        with icon_col2:
            # 🗑️ 清空配置图标按钮（动态生成tooltip）
            clear_help = get_text("clear_config_help", lang) if lang else "清空所有配置\nClear all saved configurations"
            
            if st.button("🗑️", help=clear_help, key="clear_btn", use_container_width=False, type="secondary"):
                # 保存当前语言设置
                current_language = st.session_state.get('language', 'zh-CN')
                
                # 删除保存文件
                state_file = _get_state_file_path()
                state_file = _maybe_migrate_state_file(state_file)
                custom_file = os.path.join(app_dir, "custom_products", "custom_products.json")
                
                try:
                    if os.path.exists(state_file):
                        os.remove(state_file)
                    if os.path.exists(custom_file):
                        os.remove(custom_file)
                    
                    # 清空session state（但保留语言设置）
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    
                    # 恢复语言设置
                    st.session_state['language'] = current_language
                    
                    # 重新保存语言设置
                    save_app_state()
                    
                    st.rerun()
                except Exception as e:
                    st.error(get_text("clear_config_error", lang, str(e)))
    
    # 🔄 加载之前保存的状态（首次运行时）
    if 'state_loaded' not in st.session_state:
        saved_state = load_app_state()
        if saved_state:
            st.session_state['orders'] = saved_state.get('orders', [])
            st.session_state['model_path'] = saved_state.get('model_path', '')
            st.session_state['model_loaded'] = saved_state.get('model_loaded', False)
            
            # 恢复订单配置界面的 UI 设置（随机订单生成参数等）
            ui_config = saved_state.get('ui_config', {})
            if ui_config:
                st.session_state['config_method_key'] = ui_config.get('config_method', 'random')
                # 注意：这里只在 session_state 中不存在时才设置，避免与控件的 key 参数冲突
                if 'num_orders' not in st.session_state:
                    st.session_state['num_orders'] = ui_config.get('num_orders', 5)
                if 'qty_min' not in st.session_state:
                    st.session_state['qty_min'] = ui_config.get('min_quantity', 3)
                if 'qty_max' not in st.session_state:
                    st.session_state['qty_max'] = ui_config.get('max_quantity', 10)
                if 'due_min' not in st.session_state:
                    st.session_state['due_min'] = ui_config.get('min_due', 200)
                if 'due_max' not in st.session_state:
                    st.session_state['due_max'] = ui_config.get('max_due', 700)
                if 'arrival_min' not in st.session_state:
                    st.session_state['arrival_min'] = ui_config.get('min_arrival', 0)
                if 'arrival_max' not in st.session_state:
                    st.session_state['arrival_max'] = ui_config.get('max_arrival', 50)
                if 'max_steps_single' not in st.session_state:
                    st.session_state['max_steps_single'] = ui_config.get('max_steps_single', 1500)
                # 恢复动态环境配置开关
                if 'enable_failure' not in st.session_state:
                    st.session_state['enable_failure'] = ui_config.get('enable_failure', False)
                if 'enable_emergency' not in st.session_state:
                    st.session_state['enable_emergency'] = ui_config.get('enable_emergency', False)
                # 恢复启发式算法对比选项
                if 'compare_heuristics' not in st.session_state:
                    st.session_state['compare_heuristics'] = ui_config.get('compare_heuristics', True)
            else:
                # 如果没有保存的 ui_config，初始化默认值
                if 'num_orders' not in st.session_state:
                    st.session_state['num_orders'] = 5
                if 'qty_min' not in st.session_state:
                    st.session_state['qty_min'] = 3
                if 'qty_max' not in st.session_state:
                    st.session_state['qty_max'] = 10
                if 'due_min' not in st.session_state:
                    st.session_state['due_min'] = 200
                if 'due_max' not in st.session_state:
                    st.session_state['due_max'] = 700
                if 'arrival_min' not in st.session_state:
                    st.session_state['arrival_min'] = 0
                if 'arrival_max' not in st.session_state:
                    st.session_state['arrival_max'] = 50
                if 'max_steps_single' not in st.session_state:
                    st.session_state['max_steps_single'] = 1500
                if 'enable_failure' not in st.session_state:
                    st.session_state['enable_failure'] = False
                if 'enable_emergency' not in st.session_state:
                    st.session_state['enable_emergency'] = False
                if 'compare_heuristics' not in st.session_state:
                    st.session_state['compare_heuristics'] = True
            
            # 恢复仿真结果
            last_sim = saved_state.get('last_simulation', {})
            if last_sim.get('stats'):
                st.session_state['last_stats'] = last_sim.get('stats')
                st.session_state['last_gantt_history'] = last_sim.get('gantt_history')
                st.session_state['last_score'] = last_sim.get('score')
                st.session_state['last_total_reward'] = last_sim.get('total_reward')
                
                # 恢复启发式算法对比结果
                if last_sim.get('heuristic_results'):
                    st.session_state['heuristic_results'] = last_sim.get('heuristic_results')
                
                # 同时设置到当前结果变量中，以便显示
                st.session_state['final_stats'] = last_sim.get('stats')
                st.session_state['gantt_history'] = last_sim.get('gantt_history')
                st.session_state['score'] = last_sim.get('score')
                st.session_state['total_reward'] = last_sim.get('total_reward')
                st.session_state['show_results'] = True
                
            # 如果有保存的模型路径，尝试重新加载模型
            if saved_state.get('model_loaded') and saved_state.get('model_path'):
                try:
                    model, meta_data, msg = load_model(saved_state['model_path'])
                    if model is not None:
                        st.session_state['actor_model'] = model
                        st.session_state['model_meta'] = meta_data
                except:
                    pass
        
        st.session_state['state_loaded'] = True
    
    # 步骤1：模型加载
    st.header(get_text("system_config", lang))
    
    # 模型加载方式选择
    model_input_method = st.radio(
        get_text("model_loading_method", lang),
        [get_text("from_history", lang), get_text("upload_local_model", lang)],
        horizontal=True
    )
    
    actor_model = None

    # 为不同的加载方式设置统一的行标题，让控件处于同一视觉行
    if model_input_method == get_text("from_history", lang):
        model_row_label = get_text("select_model", lang)
    else:
        model_row_label = get_text("upload_local_model_label", lang)

    st.write(model_row_label)

    col1, col2 = st.columns([3, 1])

    with col1:
        if model_input_method == get_text("from_history", lang):
            available_models = find_available_models()
            
            if not available_models:
                st.warning(get_text("no_model_found", lang))
                model_path = None
            else:
                model_options = [m["name"] for m in available_models]
                
                # 🎯 确定默认选中的模型索引（优先选择之前保存的，否则选择最新的）
                default_index = 0  # 默认选择第一个（最新的）
                saved_model_path = st.session_state.get('model_path', '')
                if saved_model_path:
                    # 查找是否有匹配的模型
                    for idx, model in enumerate(available_models):
                        if model["path"] == saved_model_path:
                            default_index = idx
                            break
                
                selected_model = st.selectbox(
                    "",
                    options=model_options,
                    index=default_index,
                    help=get_text("model_help", lang),
                    label_visibility="collapsed"
                )
                
                selected_model_info = next(m for m in available_models if m["name"] == selected_model)
                model_path = selected_model_info["path"]
                
                st.caption(f"{get_text('model_path', lang)}{model_path}")
        else:
            # 加载服务器已有模型或上传新模型
            # 1) 先列出 uploaded_models 目录下已有模型，供快速选择
            upload_dir = os.path.join(app_dir, "uploaded_models")
            existing_models = []
            if os.path.exists(upload_dir):
                for fname in sorted(os.listdir(upload_dir), reverse=True):
                    if fname.endswith(".h5") or fname.endswith(".keras"):
                        existing_models.append({
                            "name": fname,
                            "path": os.path.join(upload_dir, fname),
                        })

            model_path = None

            if existing_models:
                existing_options = [m["name"] for m in existing_models]

                # 默认优先选中当前 session_state 中的 model_path 对应的文件
                default_existing_idx = 0
                saved_model_path = st.session_state.get('model_path', '')
                if saved_model_path:
                    for idx, m in enumerate(existing_models):
                        if m["path"] == saved_model_path:
                            default_existing_idx = idx
                            break

                selected_existing = st.selectbox(
                    get_text("select_model", lang),
                    options=existing_options,
                    index=default_existing_idx,
                    help=get_text("upload_local_model_help", lang),
                )

                existing_info = next(m for m in existing_models if m["name"] == selected_existing)
                model_path = existing_info["path"]
                st.caption(f"{get_text('model_path', lang)}{model_path}")

            # 2) 允许上传新模型文件
            uploaded_file = st.file_uploader(
                get_text("upload_local_model_label", lang),
                type=["h5", "keras"],
                help=get_text("upload_local_model_help", lang),
                label_visibility="collapsed"
            )

            if uploaded_file is not None:
                os.makedirs(upload_dir, exist_ok=True)

                # 使用原始文件名，前面加上时间戳避免覆盖
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_name = uploaded_file.name.replace(" ", "_")
                saved_path = os.path.join(upload_dir, f"{timestamp}_{safe_name}")

                with open(saved_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                model_path = saved_path
                # 立即写入 session_state，便于后续状态保存与展示
                st.session_state['model_path'] = model_path

                st.caption(f"{get_text('uploaded_model_path', lang)}{model_path}")
    
    with col2:
        # 加载模型按钮
        if st.button(get_text("load_model", lang), type="primary", use_container_width=True):
            if model_path:
                with st.spinner(get_text("loading_model", lang)):
                    actor_model, meta_data, message = load_model(model_path)
                    if actor_model is not None:
                        st.session_state['actor_model'] = actor_model
                        st.session_state['model_path'] = model_path
                        st.session_state['model_meta'] = meta_data
                        st.session_state['model_loaded'] = True
                        save_app_state()  # 💾 保存状态
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
            else:
                st.error(get_text("select_model_first", lang))
    
    # 显示已加载的模型状态
    if 'actor_model' in st.session_state:
        st.success(f"{get_text('model_loaded', lang)}{st.session_state.get('model_path', 'Unknown')}")
        
        # 显示模型训练配置信息
        meta_data = st.session_state.get('model_meta')
        if (not meta_data) or (not isinstance(meta_data, dict)):
            st.info("模型元数据(meta.json)未找到，无法展示训练配置 / model meta.json not found, training config is unavailable")

        # 兼容：meta.json 可能没有 environment_config（仅保存网络/空间信息）
        env_config = {}
        if isinstance(meta_data, dict):
            env_config = meta_data.get('environment_config') or meta_data.get('env_config') or {}

        if isinstance(meta_data, dict) and (not env_config):
            st.warning("meta.json 未包含 environment_config（工作站/仿真时长等），因此无法对比训练环境与当前环境；仅展示网络与动作空间信息")

        # 即使没有环境配置，也至少展示一次模型训练配置面板（网络结构/动作空间/版本等）
        if isinstance(meta_data, dict):
            with st.expander("🔧 " + get_text("model_training_config", lang), expanded=False):
                # 基础信息卡片
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        label="🤖 " + get_text('num_agents', lang),
                        value=meta_data.get('num_agents', 'N/A')
                    )
                with col2:
                    st.metric(
                        label="🧠 TensorFlow",
                        value=meta_data.get('tensorflow_version', 'N/A')
                    )
                with col3:
                    st.metric(
                        label="📊 State Dim",
                        value=meta_data.get('state_dim', 'N/A')
                    )

                st.divider()

                # 动作空间信息
                try:
                    action_space = meta_data.get('action_space') or {}
                    if action_space:
                        st.markdown("### 🎯 动作空间配置")
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.markdown(f"**类型**: `{action_space.get('type', 'N/A')}`")
                            if action_space.get('type') == 'MultiDiscrete':
                                nvec = action_space.get('nvec', [])
                                st.markdown(f"**维度数**: {len(nvec)}")
                                st.markdown(f"**每维大小**: {nvec}")
                            else:
                                st.markdown(f"**动作数**: {action_space.get('n', 'N/A')}")
                        with col2:
                            if action_space.get('type') == 'MultiDiscrete':
                                nvec = action_space.get('nvec', [])
                                if nvec:
                                    import plotly.graph_objects as go
                                    fig = go.Figure(data=[
                                        go.Bar(
                                            x=[f"Agent {i+1}" for i in range(len(nvec))],
                                            y=nvec,
                                            text=nvec,
                                            textposition='auto',
                                            marker_color='lightblue'
                                        )
                                    ])
                                    fig.update_layout(
                                        title="各智能体动作空间大小",
                                        height=300,
                                        showlegend=False
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    pass

                st.divider()

                # 网络配置
                try:
                    net_cfg = meta_data.get('network_config') or {}
                    if net_cfg:
                        st.markdown("### 🧠 网络架构配置")
                        
                        # 网络结构
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**🏗️ 网络结构**")
                            hidden_sizes = net_cfg.get('hidden_sizes', [])
                            if hidden_sizes:
                                st.markdown(f"- 隐藏层: {' → '.join(map(str, hidden_sizes))}")
                            st.markdown(f"- Dropout率: {net_cfg.get('dropout_rate', 'N/A')}")
                            
                        with col2:
                            st.markdown("**⚙️ PPO参数**")
                            st.markdown(f"- Clip比率: {net_cfg.get('clip_ratio', 'N/A')}")
                            st.markdown(f"- 熵系数: {net_cfg.get('entropy_coeff', 'N/A')}")
                            st.markdown(f"- PPO轮数: {net_cfg.get('ppo_epochs', 'N/A')}")
                            st.markdown(f"- 小批次数: {net_cfg.get('num_minibatches', 'N/A')}")

                        # 高级配置
                        with st.expander("🔬 高级配置", expanded=False):
                            adv_col1, adv_col2 = st.columns(2)
                            with adv_col1:
                                st.markdown("**梯度与优势**")
                                st.markdown(f"- 梯度裁剪: {net_cfg.get('grad_clip_norm', 'N/A')}")
                                st.markdown(f"- 优势裁剪: {net_cfg.get('advantage_clip_val', 'N/A')}")
                                st.markdown(f"- Gamma: {net_cfg.get('gamma', 'N/A')}")
                                st.markdown(f"- Lambda GAE: {net_cfg.get('lambda_gae', 'N/A')}")
                            
                            with adv_col2:
                                st.markdown("**启发式混合**")
                                hm_enabled = net_cfg.get('heuristic_mixture_enabled', False)
                                st.markdown(f"- 启用: {'✅' if hm_enabled else '❌'}")
                                if hm_enabled:
                                    st.markdown(f"- Beta: {net_cfg.get('heuristic_mixture_beta', 'N/A')}")
                                
                                st.markdown("**教师网络**")
                                bc_enabled = net_cfg.get('teacher_bc_enabled', False)
                                st.markdown(f"- 启用: {'✅' if bc_enabled else '❌'}")
                                if bc_enabled:
                                    st.markdown(f"- 模式: {net_cfg.get('teacher_bc_mode', 'N/A')}")
                except Exception:
                    pass

                # 环境配置（如果存在）
                if env_config:
                    st.divider()
                    st.markdown("### ⚙️ 环境配置")
                    
                    # 显示工作站配置
                    st.markdown(f"**{get_text('workstation_config', lang)}**")
                    workstations_df = pd.DataFrame([
                        {
                            get_text('workstation_name', lang): get_text(ws_name, lang),
                            get_text('equipment_count', lang): ws_config['count'],
                            get_text('capacity', lang): ws_config['capacity']
                        }
                        for ws_name, ws_config in env_config.get('workstations', {}).items()
                    ])
                    st.dataframe(workstations_df, use_container_width=True, hide_index=True)
                    
                    # 显示其他环境配置信息
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            get_text('simulation_time', lang),
                            f"{env_config.get('simulation_time', 'N/A')} {env_config.get('time_unit', '')}"
                        )
                    with col2:
                        st.metric(
                            get_text('num_product_types', lang),
                            env_config.get('num_product_types', 'N/A')
                        )
                    with col3:
                        pass  # 预留位置

                # 保存时间
                if 'save_timestamp' in meta_data:
                    try:
                        timestamp = datetime.fromisoformat(meta_data['save_timestamp'])
                        st.divider()
                        st.caption(f"💾 {get_text('model_save_time', lang)}{timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                    except:
                        pass

        if isinstance(meta_data, dict) and env_config:
            trained_ws = env_config.get('workstations') or {}

            current_ws = WORKSTATIONS or {}
            mismatch_details = {
                'missing_in_current': [],
                'extra_in_current': [],
                'changed': []
            }

            try:
                trained_keys = set(trained_ws.keys())
                current_keys = set(current_ws.keys())

                mismatch_details['missing_in_current'] = sorted(list(trained_keys - current_keys))
                mismatch_details['extra_in_current'] = sorted(list(current_keys - trained_keys))

                for k in sorted(list(trained_keys & current_keys)):
                    tcfg = trained_ws.get(k) or {}
                    ccfg = current_ws.get(k) or {}
                    t_count = int(tcfg.get('count', 0))
                    c_count = int(ccfg.get('count', 0))
                    t_cap = int(tcfg.get('capacity', 0))
                    c_cap = int(ccfg.get('capacity', 0))
                    if (t_count != c_count) or (t_cap != c_cap):
                        mismatch_details['changed'].append({
                            'name': k,
                            'trained': {'count': t_count, 'capacity': t_cap},
                            'current': {'count': c_count, 'capacity': c_cap}
                        })
            except Exception:
                mismatch_details = None

            time_mismatch = False
            try:
                trained_sim_time = env_config.get('simulation_time', None)
                if trained_sim_time is not None:
                    time_mismatch = float(trained_sim_time) != float(SIMULATION_TIME)
            except Exception:
                time_mismatch = False

            has_ws_mismatch = bool(mismatch_details) and (
                bool(mismatch_details.get('missing_in_current')) or
                bool(mismatch_details.get('extra_in_current')) or
                bool(mismatch_details.get('changed'))
            )

            if has_ws_mismatch or bool(time_mismatch):
                st.warning(get_text('model_env_config_mismatch_warning', lang))
                with st.expander(get_text('model_env_config_mismatch_details', lang), expanded=False):
                    if bool(time_mismatch):
                        _time_unit = globals().get('TIME_UNIT', '')
                        st.markdown(
                            f"**{get_text('simulation_time', lang)}**: "
                            f"{get_text('trained_config', lang)} {env_config.get('simulation_time', 'N/A')} {env_config.get('time_unit', '')} / "
                            f"{get_text('current_config', lang)} {SIMULATION_TIME} {_time_unit}"
                        )

                    if mismatch_details:
                        if mismatch_details.get('missing_in_current'):
                            st.markdown(f"**{get_text('missing_workstations_in_current', lang)}**")
                            st.write([get_text(x, lang) for x in mismatch_details['missing_in_current']])

                        if mismatch_details.get('extra_in_current'):
                            st.markdown(f"**{get_text('extra_workstations_in_current', lang)}**")
                            st.write([get_text(x, lang) for x in mismatch_details['extra_in_current']])

                        if mismatch_details.get('changed'):
                            st.markdown(f"**{get_text('changed_workstations', lang)}**")
                            changed_rows = []
                            for item in mismatch_details['changed']:
                                name = item.get('name')
                                t = item.get('trained') or {}
                                c = item.get('current') or {}
                                changed_rows.append({
                                    get_text('workstation_name', lang): get_text(name, lang),
                                    get_text('trained_config', lang): f"count={t.get('count')}, capacity={t.get('capacity')}",
                                    get_text('current_config', lang): f"count={c.get('count')}, capacity={c.get('capacity')}"
                                })
                            st.dataframe(pd.DataFrame(changed_rows), use_container_width=True, hide_index=True)
    
    # 自定义产品工艺路线管理（系统配置的一部分）
    with st.expander(get_text("custom_products", lang), expanded=False):
        st.caption(get_text("custom_products_caption", lang))
        
        # 初始化自定义产品路线
        if 'custom_products' not in st.session_state:
            st.session_state['custom_products'] = load_custom_products()
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            new_product_name = st.text_input(get_text("new_product_name", lang), placeholder=get_text("new_product_placeholder", lang))
        
        with col2:
            st.write("")  # 空行对齐
        
        st.write(get_text("process_route_definition", lang))
        
        # 显示可用工作站
        st.caption(get_text("available_workstations", lang, ', '.join([get_text(ws, lang) for ws in WORKSTATIONS.keys()])))
        
        # 工艺步骤输入
        num_steps = st.number_input(get_text("num_steps", lang), min_value=1, max_value=10, value=3, key="custom_steps")
        
        route_steps = []
        for i in range(num_steps):
            col1, col2 = st.columns([2, 1])
            with col1:
                station = st.selectbox(
                    f'{get_text("step_label", lang)} {i+1} - {get_text("workstation_label", lang)}',
                    options=list(WORKSTATIONS.keys()),
                    format_func=lambda x: get_text(x, lang),
                    key=f"custom_station_{i}"
                )
            with col2:
                time = st.number_input(
                    f'{get_text("step_label", lang)} {i+1} - {get_text("time_label", lang)}',
                    min_value=1,
                    max_value=100,
                    value=10,
                    key=f"custom_time_{i}"
                )
            route_steps.append({"station": station, "time": time})
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button(get_text("add_product", lang)):
                if new_product_name:
                    if new_product_name in PRODUCT_ROUTES:
                        st.error(get_text("error_product_already_exists_system", lang, new_product_name))
                    else:
                        st.session_state['custom_products'][new_product_name] = route_steps
                        save_custom_products(st.session_state['custom_products'])
                        st.success(get_text("success_product_added", lang, new_product_name))
                        st.rerun()
                else:
                    st.error(get_text("error_enter_product_name", lang))
        
        # 显示已添加的自定义产品
        if st.session_state['custom_products']:
            st.divider()
            st.write(get_text("added_custom_products", lang))
            
            for prod_name, route in st.session_state['custom_products'].items():
                col1, col2 = st.columns([4, 1])
                with col1:
                    route_str = " → ".join([f"{s['station']}({s['time']}{get_text('minutes', lang).strip()})" for s in route])
                    st.text(f"• {prod_name}: {route_str}")
                with col2:
                    if st.button("🗑️", key=f"del_{prod_name}"):
                        del st.session_state['custom_products'][prod_name]
                        save_custom_products(st.session_state['custom_products'])
                        st.rerun()
    
    st.divider()
    
    # 步骤2：订单配置
    st.header(get_text("order_config", lang))
    
    # 提供两种配置方式
    saved_method_key = st.session_state.get('config_method_key', 'random')
    method_options = [get_text("random_orders", lang), get_text("custom_orders", lang)]
    default_index = 1 if saved_method_key == 'custom' else 0

    config_method = st.radio(
        get_text("choose_config_method", lang),
        method_options,
        horizontal=True,
        label_visibility="collapsed",
        index=default_index
    )

    # 将当前选择映射回内部 key 以便持久化
    if config_method == get_text("custom_orders", lang):
        st.session_state['config_method_key'] = 'custom'
    else:
        st.session_state['config_method_key'] = 'random'
    
    if config_method == get_text("custom_orders", lang):
        # 初始化订单列表
        if 'orders' not in st.session_state:
            st.session_state['orders'] = []
        
        st.subheader(get_text("add_order", lang))
        
        # 合并系统产品和自定义产品
        custom_products = st.session_state.get('custom_products', {})
        all_products = list(PRODUCT_ROUTES.keys()) + list(custom_products.keys())
        
        # 添加订单表单
        with st.form("add_order_form"):
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                product = st.selectbox(
                    get_text("product_type", lang),
                    options=all_products
                )
            
            with col2:
                quantity = st.number_input(
                    get_text("quantity", lang),
                    min_value=1,
                    max_value=100,
                    value=5
                )
            
            with col3:
                priority = st.number_input(
                    get_text("priority", lang),
                    min_value=1,
                    max_value=5,
                    value=1,
                    help=get_text("priority_help", lang)
                )
            
            with col4:
                arrival_time = st.number_input(
                    get_text("arrival_time", lang),
                    min_value=0,
                    max_value=500,
                    value=0,
                    step=10,
                    help=get_text("arrival_time_help", lang)
                )
            
            with col5:
                due_date = st.number_input(
                    get_text("due_date", lang),
                    min_value=60,
                    max_value=2000,
                    value=300,
                    step=10
                )

            btn_col1, btn_col2 = st.columns([1, 1])
            with btn_col1:
                submitted = st.form_submit_button(get_text("add_order_button", lang))
            with btn_col2:
                import_submitted = st.form_submit_button("从JSON导入")

            st.markdown("粘贴JSON订单配置（支持仿真结果JSON的 orders 字段，或导出配置的订单list）")
            import_json_text = st.text_area(
                "",
                height=150,
                key="orders_import_text",
                label_visibility="collapsed"
            )

            if submitted:
                order = {
                    "product": product,
                    "quantity": int(quantity),
                    "priority": int(priority),
                    "arrival_time": int(arrival_time),
                    "due_date": int(due_date)
                }
                st.session_state['orders'].append(order)
                clear_simulation_results()  # 🔧 清空之前的调度结果
                save_app_state()  # 💾 保存状态
                st.success(get_text("order_added_full", lang, product, quantity, arrival_time, due_date))
                st.rerun()

            if import_submitted:
                if not (import_json_text or "").strip():
                    st.warning("请先粘贴JSON内容")
                else:
                    try:
                        obj = json.loads(import_json_text)
                        extracted = extract_orders_from_json_obj(obj)
                        normalized = normalize_orders_list(extracted)
                        if not normalized:
                            st.error("JSON格式不正确：需要是订单列表，或包含 orders 字段的仿真结果JSON")
                        else:
                            st.session_state['orders'] = normalized
                            clear_simulation_results()
                            save_app_state()
                            st.success(f"已导入 {len(normalized)} 条订单")
                            st.rerun()
                    except Exception as e:
                        st.error(f"解析JSON失败：{str(e)}")
    
    else:  # 随机生成订单
        st.subheader(get_text("random_order_gen", lang))
        
        # 确保所有 UI 参数在 session_state 中有默认值
        if 'num_orders' not in st.session_state:
            st.session_state['num_orders'] = 5
        if 'qty_min' not in st.session_state:
            st.session_state['qty_min'] = 3
        if 'qty_max' not in st.session_state:
            st.session_state['qty_max'] = 10
        if 'due_min' not in st.session_state:
            st.session_state['due_min'] = 200
        if 'due_max' not in st.session_state:
            st.session_state['due_max'] = 700
        if 'arrival_min' not in st.session_state:
            st.session_state['arrival_min'] = 0
        if 'arrival_max' not in st.session_state:
            st.session_state['arrival_max'] = 50
        
        # 订单数量 - 只使用 key，不设置 value
        num_orders = st.slider(
            get_text("order_count", lang),
            min_value=3,
            max_value=10,
            key="num_orders"
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write(get_text("product_quantity_range", lang))
            subcol1, subcol2 = st.columns(2)
            with subcol1:
                min_quantity = st.number_input(get_text("from", lang), min_value=1, max_value=50, key="qty_min")
            with subcol2:
                max_quantity = st.number_input(get_text("to", lang), min_value=1, max_value=50, key="qty_max")
        
        with col2:
            st.write(get_text("due_date_range", lang))
            subcol1, subcol2 = st.columns(2)
            with subcol1:
                min_due = st.number_input(get_text("from", lang), min_value=100, max_value=2000, step=10, key="due_min")
            with subcol2:
                max_due = st.number_input(get_text("to", lang), min_value=100, max_value=2000, step=10, key="due_max")
        
        with col3:
            st.write(get_text("arrival_time_range", lang))
            subcol1, subcol2 = st.columns(2)
            with subcol1:
                min_arrival = st.number_input(get_text("from", lang), min_value=0, max_value=500, step=10, key="arrival_min")
            with subcol2:
                max_arrival = st.number_input(get_text("to", lang), min_value=0, max_value=500, step=10, key="arrival_max")
        
        if st.button(get_text("generate_random", lang), type="primary"):
            # 🔧 支持自定义产品：合并系统产品和自定义产品
            import random
            custom_products = st.session_state.get('custom_products', {})
            all_products = list(PRODUCT_ROUTES.keys()) + list(custom_products.keys())
            
            # 手动生成随机订单（包含自定义产品和到达时间范围）
            random_orders = []
            for i in range(num_orders):
                product = random.choice(all_products)
                quantity = random.randint(min_quantity, max_quantity)
                priority = random.choices([1, 2, 3], weights=[0.3, 0.5, 0.2])[0]
                # 确保时间值是10的倍数
                arrival_time = round(random.uniform(min_arrival, max_arrival) / 10) * 10
                due_date = round(random.uniform(min_due, max_due) / 10) * 10
                
                random_orders.append({
                    "product": product,
                    "quantity": quantity,
                    "priority": priority,
                    "arrival_time": arrival_time,
                    "due_date": due_date
                })
            
            st.session_state['orders'] = random_orders
            clear_simulation_results()  # 🔧 清空之前的调度结果
            save_app_state()  # 💾 保存状态
            st.success(get_text("random_generated", lang, len(random_orders)))
            st.rerun()
    
    # 显示当前订单列表（所有模式通用）
    if st.session_state.get('orders'):
        st.divider()
        st.subheader(get_text("current_orders", lang))
        
        orders_df = pd.DataFrame(st.session_state['orders'])
        orders_df.index = range(1, len(orders_df) + 1)
        
        # 根据列数设置列名
        if len(orders_df.columns) == 5:
            orders_df = orders_df[['product', 'quantity', 'priority', 'arrival_time', 'due_date']]
            orders_df.columns = eval(get_text("order_list_columns_5", lang))
        else:
            orders_df.columns = eval(get_text("order_list_columns_4", lang))
        
        st.dataframe(orders_df, use_container_width=True)
        
        # 订单管理按钮
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            if st.button(get_text("clear_orders", lang)):
                st.session_state['orders'] = []
                clear_simulation_results()  # 🔧 清空之前的调度结果
                save_app_state()  # 💾 保存状态
                st.rerun()
        
        with col2:
            config_json = json.dumps(st.session_state['orders'], indent=2, ensure_ascii=False)
            st.download_button(
                label=get_text("export_config", lang),
                data=config_json,
                file_name=f"orders_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        # 显示订单统计
        total_parts = sum(order['quantity'] for order in st.session_state['orders'])
        st.caption(get_text("order_summary", lang, len(st.session_state['orders']), total_parts))
        
        # 🔧 新增：订单配置合理性检测
        st.divider()
        st.subheader(get_text("order_analysis", lang))
        
        custom_products = st.session_state.get('custom_products', {})
        validation_result = validate_order_config(st.session_state['orders'], custom_products)
        
        if not validation_result['valid']:
            st.error(get_text("config_invalid", lang))
            for warning in validation_result['warnings']:
                st.warning(warning)
        else:
            info = validation_result['info']
            
            # 显示难度评估
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(get_text("difficulty", lang), validation_result['difficulty_level'])
            with col2:
                st.metric(get_text("total_products", lang), f"{info['total_parts']}")
            with col3:
                st.metric(get_text("theory_time", lang), f"{info['theoretical_makespan']:.0f}{get_text('minutes', lang)}")
            with col4:
                st.metric(get_text("bottleneck", lang), get_text(info['bottleneck_station'], lang))
            
            # 显示详细信息和警告
            with st.expander(get_text("view_details", lang), expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(get_text("basic_stats", lang))
                    st.write(get_text("total_processing_time_label", lang, info['total_processing_time']))
                    st.write(get_text("avg_due_date_label", lang, info['avg_due_date']))
                    st.write(get_text("min_due_date_label", lang, info['min_due_date']))
                    st.write(get_text("max_due_date_label", lang, info['max_due_date']))
                    if info.get('has_arrival_time'):
                        st.write(get_text("max_arrival_time_label", lang, info['max_arrival_time']))
                
                with col2:
                    st.write(get_text("bottleneck_analysis", lang))
                    st.write(get_text("bottleneck_station_label", lang, get_text(info['bottleneck_station'], lang)))
                    st.write(get_text("bottleneck_load_label", lang, info['bottleneck_load']))
                    st.write(get_text("load_ratio_label", lang, info['bottleneck_load']/SIMULATION_TIME*100))
                    st.write(get_text("standard_simulation_time_label", lang, SIMULATION_TIME))
                
                # 显示警告和建议
                if validation_result['warnings']:
                    st.write(get_text("tips_and_suggestions", lang))
                    for warning in validation_result['warnings']:
                        st.write(f"- {warning}")
    
    # 开始调度按钮和结果展示区域
    st.divider()
    
    if 'actor_model' not in st.session_state:
        st.warning(get_text("load_model_first", lang))
    elif not st.session_state.get('orders', []):
        st.warning(get_text("config_orders_first", lang))
    else:
        # 添加启发式算法对比选项
        st.subheader(get_text("comparison_options", lang))
        compare_heuristics = st.checkbox(
            get_text("compare_heuristics_checkbox", lang),
            value=st.session_state.get('compare_heuristics', True),
            help=get_text("compare_heuristics_help", lang),
            key="compare_heuristics"
        )

        base_seed_single = st.number_input(
            get_text("base_seed_single", lang),
            min_value=0,
            max_value=1000000,
            value=int(st.session_state.get('base_seed_single', 42)),
            step=1,
            help=get_text("base_seed_single_help", lang),
            key="base_seed_single"
        )
        
        # 12-02 新增：动态环境配置
        st.subheader(get_text("dynamic_env_config", lang))
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            enable_failure = st.checkbox(
                get_text("enable_failure_sim", lang), 
                value=st.session_state.get('enable_failure', False), 
                help=get_text("enable_failure_sim_help", lang),
                key="enable_failure"
            )
        with col_d2:
            enable_emergency = st.checkbox(
                get_text("enable_emergency_sim", lang), 
                value=st.session_state.get('enable_emergency', False), 
                help=get_text("enable_emergency_sim_help", lang),
                key="enable_emergency"
            )

        # 设备故障高级参数配置
        failure_config = {}
        if enable_failure:
            # 初始化expander展开状态
            if 'failure_expander_expanded' not in st.session_state:
                st.session_state['failure_expander_expanded'] = True
            
            with st.expander(get_text("failure_params_expander", lang), expanded=st.session_state['failure_expander_expanded']):
                st.markdown(get_text("failure_params_desc", lang))
                
                fcol1, fcol2, fcol3 = st.columns(3)
                with fcol1:
                    mtbf_hours = st.number_input(
                        get_text("mtbf_hours", lang),
                        min_value=10.0,
                        max_value=60.0,
                        value=st.session_state.get('mtbf_hours', 24.0),
                        step=2.0,
                        help=get_text("mtbf_hours_help", lang),
                        key="mtbf_hours"
                    )
                with fcol2:
                    mttr_minutes = st.number_input(
                        get_text("mttr_minutes", lang),
                        min_value=10.0,
                        max_value=120.0,
                        value=st.session_state.get('mttr_minutes', 30.0),
                        step=5.0,
                        help=get_text("mttr_minutes_help", lang),
                        key="mttr_minutes"
                    )
                with fcol3:
                    failure_prob = st.number_input(
                        get_text("failure_probability", lang),
                        min_value=0.01,
                        max_value=0.10,
                        value=st.session_state.get('failure_prob', 0.02),
                        step=0.01,
                        format="%.2f",
                        help=get_text("failure_probability_help", lang),
                        key="failure_prob"
                    )
                
                failure_config = {
                    'mtbf_hours': mtbf_hours,
                    'mttr_minutes': mttr_minutes,
                    'failure_probability': failure_prob
                }

                st.session_state['failure_config'] = failure_config
        
        # 紧急插单高级参数配置
        emergency_config = {}
        if enable_emergency:
            # 初始化expander展开状态
            if 'emergency_expander_expanded' not in st.session_state:
                st.session_state['emergency_expander_expanded'] = True
            
            with st.expander(get_text("emergency_params_expander", lang), expanded=st.session_state['emergency_expander_expanded']):
                st.markdown(get_text("emergency_params_desc", lang))
                
                ecol1, ecol2, ecol3 = st.columns(3)
                with ecol1:
                    arrival_rate = st.number_input(
                        get_text("emergency_arrival_rate", lang),
                        min_value=0.05,
                        max_value=0.5,
                        value=st.session_state.get('arrival_rate', 0.1),
                        step=0.05,
                        help=get_text("emergency_arrival_rate_help", lang),
                        key="arrival_rate"
                    )
                with ecol2:
                    priority_boost = st.number_input(
                        get_text("emergency_priority_boost", lang),
                        min_value=0,
                        max_value=3,
                        value=st.session_state.get('priority_boost', 0),
                        step=1,
                        help=get_text("emergency_priority_boost_help", lang),
                        key="priority_boost"
                    )
                with ecol3:
                    due_reduction = st.slider(
                        get_text("emergency_due_date_reduction", lang),
                        min_value=0.4,
                        max_value=0.85,
                        value=st.session_state.get('due_reduction', 0.7),
                        step=0.05,
                        help=get_text("emergency_due_date_reduction_help", lang),
                        key="due_reduction"
                    )
                
                emergency_config = {
                    'arrival_rate': arrival_rate,
                    'priority_boost': priority_boost,
                    'due_date_reduction': due_reduction
                }

                st.session_state['emergency_config'] = emergency_config

        max_steps_single = st.number_input(
            get_text("max_steps", lang),
            min_value=500,
            max_value=20000,
            value=int(st.session_state.get('max_steps_single', 1500)),
            step=100,
            help=get_text("max_steps_help", lang),
            key="max_steps_single"
        )
        st.caption(get_text("max_steps_comparison_help", lang))

        st.write("")  # 空行
        
        if st.button(get_text("start_simulation", lang), type="primary", use_container_width=True):
            try:
                actor_model = st.session_state['actor_model']
                orders = st.session_state['orders']
                custom_products = st.session_state.get('custom_products', {})
                
                # 验证动态事件配置的完整性
                if enable_failure and (not failure_config or len(failure_config) == 0):
                    st.error("设备故障已启用但配置不完整，请检查配置参数")
                    st.stop()
                
                if enable_emergency and (not emergency_config or len(emergency_config) == 0):
                    st.error("紧急插单已启用但配置不完整，请检查配置参数")
                    st.stop()
                
                # 创建进度条和状态文本
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # 运行MARL模型
                status_text.text(get_text("running_marl", lang))
                final_stats, gantt_history, score, total_reward = run_scheduling(
                    actor_model, orders, custom_products, 
                    max_steps=int(max_steps_single),
                    progress_bar=progress_bar, 
                    status_text=status_text,
                    enable_failure=enable_failure,
                    enable_emergency=enable_emergency,
                    failure_config=failure_config or {},
                    emergency_config=emergency_config or {},
                    seed=int(base_seed_single)
                )
                
                # 保存MARL结果
                st.session_state['final_stats'] = final_stats
                st.session_state['gantt_history'] = gantt_history
                st.session_state['score'] = score
                st.session_state['total_reward'] = total_reward
                st.session_state['show_results'] = True
                
                # 如果需要对比启发式算法
                heuristic_results = None
                if compare_heuristics:
                    heuristic_results = {}
                    
                    for heuristic in ['FIFO', 'EDD', 'SPT', 'ATC']:
                        status_text.text(get_text("running_heuristic", lang, heuristic))
                        progress_bar.progress(0)
                        
                        h_stats, h_history, h_score = run_heuristic_scheduling(
                            heuristic, orders, custom_products,
                            max_steps=int(max_steps_single),
                            progress_bar=progress_bar,
                            status_text=status_text,
                            enable_failure=enable_failure,
                            enable_emergency=enable_emergency,
                            failure_config=failure_config,
                            emergency_config=emergency_config,
                            seed=int(base_seed_single)
                        )
                        
                        heuristic_results[heuristic] = {
                            'stats': h_stats,
                            'history': h_history,
                            'score': h_score
                        }
                    
                    st.session_state['heuristic_results'] = heuristic_results
                else:
                    # 如果不对比，清除之前的结果
                    if 'heuristic_results' in st.session_state:
                        del st.session_state['heuristic_results']
                
                # 保存完整调度结果（含甘特图和对比数据）
                model_path = st.session_state.get('model_path', '')
                seeds_used = [int(base_seed_single)]
                save_schedule_result(
                    model_path, orders, final_stats, score, 
                    gantt_history=gantt_history,
                    heuristic_results=heuristic_results,
                    enable_failure=st.session_state.get('enable_failure', False),
                    enable_emergency=st.session_state.get('enable_emergency', False),
                    seeds_used=seeds_used
                )

                # 同时保存到持久化变量（包括启发式算法对比结果）
                st.session_state['last_stats'] = final_stats
                st.session_state['last_gantt_history'] = gantt_history
                st.session_state['last_score'] = score
                st.session_state['last_total_reward'] = total_reward
                # heuristic_results 已经在上面保存到 session_state 了
                
                save_app_state()  # 💾 保存状态
                progress_bar.empty()
                status_text.empty()
                st.success(get_text("simulation_complete", lang))
                st.rerun()
                
            except Exception as e:
                st.error(f"{get_text('simulation_failed', lang)}{str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # 显示调度结果（在按钮下方）
    if st.session_state.get('show_results', False) and 'final_stats' in st.session_state:
        st.divider()
        st.header(get_text("results", lang))
        
        stats = st.session_state['final_stats']
        gantt_history = st.session_state['gantt_history']
        score = st.session_state['score']
        total_reward = st.session_state['total_reward']
        orders = st.session_state['orders']
        
        # KPI指标展示
        st.subheader(get_text("kpi", lang))
        
        total_parts_target = sum(order["quantity"] for order in orders)
        completion_rate = (stats['total_parts'] / total_parts_target) * 100 if total_parts_target > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label=get_text("completed_products", lang),
                value=f"{stats['total_parts']}/{total_parts_target}",
                delta=f"{completion_rate:.1f}%"
            )
        
        with col2:
            st.metric(
                label=get_text("makespan", lang),
                value=f"{stats['makespan']:.1f}{get_text('minutes', lang)}"
            )
        
        with col3:
            st.metric(
                label=get_text("utilization", lang),
                value=f"{stats['mean_utilization']*100:.1f}%"
            )
        
        with col4:
            st.metric(
                label=get_text("tardiness", lang),
                value=f"{stats['total_tardiness']:.1f}{get_text('minutes', lang)}"
            )
        
        # 综合评分
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label=get_text("score", lang),
                value=f"{score:.3f}",
                help=get_text("score_help", lang)
            )
        with col2:
            st.metric(
                label=get_text("reward", lang),
                value=f"{total_reward:.1f}"
            )
        
        # 设备利用率图表
        with st.expander(get_text("util_analysis", lang), expanded=True):
            util_chart = create_utilization_chart(stats)
            if util_chart:
                st.plotly_chart(util_chart, use_container_width=True)
        
        # 甘特图（MARL）
        with st.expander(get_text("gantt_chart", lang) + " - MARL (PPO)", expanded=True):
            gantt_fig = create_gantt_chart(gantt_history, stats.get('event_timeline'))
            if gantt_fig:
                # 更新标题显示算法名称
                gantt_fig.update_layout(title=f"{get_text('gantt_chart_title', lang)} - MARL (PPO)")
                st.plotly_chart(gantt_fig, use_container_width=True)
                
                # 提供下载选项
                if st.button(get_text("download_gantt", lang), key="download_marl"):
                    html_str = gantt_fig.to_html()
                    st.download_button(
                        label=get_text("download_gantt_btn", lang),
                        data=html_str,
                        file_name=f"gantt_marl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html",
                        key="download_marl_confirm"
                    )
            else:
                st.warning(get_text("warn_gantt_no_data", lang))
        
        # 启发式算法对比结果
        if 'heuristic_results' in st.session_state:
            st.divider()
            st.subheader(get_text("algorithm_performance_comparison", lang))
            
            # 创建对比表格
            comparison_data = []
            
            # MARL结果
            marl_completion_rate = (stats['total_parts'] / total_parts_target) * 100 if total_parts_target > 0 else 0
            comparison_data.append({
                get_text("algorithm", lang): "MARL (PPO)",
                get_text("completion_rate", lang): f"{marl_completion_rate:.1f}%",
                get_text("completion_time", lang): f"{stats['makespan']:.1f}",
                get_text("avg_utilization", lang): f"{stats['mean_utilization']*100:.1f}%",
                get_text("total_delay", lang): f"{stats['total_tardiness']:.1f}",
                get_text("comprehensive_score", lang): f"{score:.3f}"
            })
            
            # 启发式算法结果
            for heuristic_name, heuristic_data in st.session_state['heuristic_results'].items():
                h_stats = heuristic_data['stats']
                h_score = heuristic_data['score']
                h_completion_rate = (h_stats['total_parts'] / total_parts_target) * 100 if total_parts_target > 0 else 0
                
                comparison_data.append({
                    get_text("algorithm", lang): heuristic_name,
                    get_text("completion_rate", lang): f"{h_completion_rate:.1f}%",
                    get_text("completion_time", lang): f"{h_stats['makespan']:.1f}",
                    get_text("avg_utilization", lang): f"{h_stats['mean_utilization']*100:.1f}%",
                    get_text("total_delay", lang): f"{h_stats['total_tardiness']:.1f}",
                    get_text("comprehensive_score", lang): f"{h_score:.3f}"
                })
            
            # 显示对比表格
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            # 显示启发式算法的甘特图
            st.divider()
            st.subheader(get_text("heuristic_gantt_comparison", lang))
            
            for heuristic_name, heuristic_data in st.session_state['heuristic_results'].items():
                h_history = heuristic_data['history']
                h_stats = heuristic_data.get('stats', {}) if isinstance(heuristic_data, dict) else {}
                
                with st.expander(get_text("gantt_chart_algorithm", lang, heuristic_name), expanded=False):
                    h_gantt_fig = create_gantt_chart(h_history, h_stats.get('event_timeline'))
                    if h_gantt_fig:
                        # 更新标题显示算法名称
                        h_gantt_fig.update_layout(title=f"{get_text('gantt_chart_title', lang)} - {heuristic_name}")
                        st.plotly_chart(h_gantt_fig, use_container_width=True)
                        
                        # 提供下载选项
                        if st.button(get_text("download_algorithm_gantt", lang, heuristic_name), key=f"download_{heuristic_name}"):
                            html_str = h_gantt_fig.to_html()
                            st.download_button(
                                label=get_text("download_algorithm_gantt_html", lang, heuristic_name),
                                data=html_str,
                                file_name=f"gantt_{heuristic_name.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                                mime="text/html",
                                key=f"download_{heuristic_name}_confirm"
                            )
                    else:
                        st.warning(get_text("no_gantt_data_algorithm", lang, heuristic_name))
        
        # 详细统计信息
        with st.expander(get_text("detailed_stats", lang)):
            st.json({
                get_text("completed_parts_json", lang): stats['total_parts'],
                get_text("makespan_json", lang): stats['makespan'],
                get_text("mean_util_json", lang): f"{stats['mean_utilization']*100:.2f}%",
                get_text("total_tardiness_json", lang): stats['total_tardiness'],
                get_text("max_tardiness_json", lang): stats.get('max_tardiness', 0),
                get_text("util_details_json", lang): {k: f"{v*100:.2f}%" for k, v in stats.get('equipment_utilization', {}).items()}
            })
    
    # ============ 模型性能对比模块 ============
    st.divider()
    st.header("🛠 " + get_text("model_comparison", lang))

    # --- 动态事件消融测试：同一模型在启用/禁用动态事件下的性能对比 ---
    with st.expander(get_text("dynamic_event_ablation_title", lang), expanded=False):
        st.markdown(get_text("dynamic_event_ablation_help", lang))

        if not st.session_state.get('orders', []):
            st.warning(get_text("config_orders_first_comparison", lang))
        else:
            ablation_models = find_available_models()
            if not ablation_models:
                st.warning(get_text("no_model_found", lang))
            else:
                ablation_model_options = {m["name"]: m["path"] for m in ablation_models}
                default_ablation_model = list(ablation_model_options.keys())[:1]
                selected_ablation_model = st.selectbox(
                    get_text("select_single_model", lang),
                    options=list(ablation_model_options.keys()),
                    index=0 if default_ablation_model else 0,
                    key="ablation_model_select"
                )

                include_heuristics_ablation = st.checkbox(
                    get_text("include_heuristics_baseline", lang),
                    value=bool(st.session_state.get('include_heuristics_ablation', True)),
                    help=get_text("include_heuristics_baseline_help", lang),
                    key="include_heuristics_ablation"
                )

                col1, col2 = st.columns(2)
                with col1:
                    ablation_max_steps = int(st.session_state.get('max_steps_single', 1500))
                    st.caption(f"{get_text('max_steps', lang)}: {int(ablation_max_steps)}")
                with col2:
                    ablation_multi_seed = st.toggle(
                        get_text("seed_mode", lang),
                        value=False,
                        help=get_text("seed_mode_help", lang),
                        key="ablation_multi_seed"
                    )

                col3, col4 = st.columns(2)
                with col3:
                    ablation_runs = st.number_input(
                        get_text("comparison_runs", lang),
                        min_value=1,
                        max_value=5,
                        value=2,
                        step=1,
                        disabled=(not ablation_multi_seed),
                        help=get_text("comparison_runs_help", lang),
                        key="ablation_runs"
                    )
                with col4:
                    ablation_base_seed = st.number_input(
                        get_text("base_seed", lang),
                        min_value=0,
                        max_value=1000000,
                        value=42,
                        step=1,
                        help=get_text("base_seed_help", lang),
                        key="ablation_base_seed"
                    )

                enabled_failure = bool(st.session_state.get('enable_failure', False))
                enabled_emergency = bool(st.session_state.get('enable_emergency', False))
                ablation_ready = bool(enabled_failure or enabled_emergency)
                if not ablation_ready:
                    st.warning(get_text("warn_select_at_least_one_dynamic_event", lang))

                if st.button(get_text("start_dynamic_event_ablation", lang), type="primary", use_container_width=True, disabled=(not ablation_ready)):
                    try:
                        # 生成 seeds_used
                        if ablation_multi_seed:
                            seeds_used = [int(ablation_base_seed + i) for i in range(int(ablation_runs))]
                        else:
                            seeds_used = [int(ablation_base_seed)]

                        # 读取当前动态事件配置作为"启用组"配置
                        enabled_failure_cfg = st.session_state.get('failure_config', {})
                        enabled_emergency_cfg = st.session_state.get('emergency_config', {})

                        model_path = ablation_model_options[selected_ablation_model]
                        actor_model, _, message = load_model(model_path)
                        if actor_model is None:
                            st.error(f"{get_text('load_model_failed', lang, selected_ablation_model)}: {message}")
                        else:
                            orders_cfg = st.session_state['orders']
                            custom_products = st.session_state.get('custom_products')

                            disabled_runs = []
                            enabled_runs = []

                            heuristic_disabled = {h: [] for h in ['FIFO', 'EDD', 'SPT', 'ATC']} if include_heuristics_ablation else {}
                            heuristic_enabled = {h: [] for h in ['FIFO', 'EDD', 'SPT', 'ATC']} if include_heuristics_ablation else {}

                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            num_heuristics = 4 if include_heuristics_ablation else 0
                            total_steps = len(seeds_used) * (2 + 2 * num_heuristics)
                            done = 0

                            def _make_step_cb(job_index: int):
                                def _cb(step_now: int, step_max: int, _seed=None, _mode=None):
                                    try:
                                        ratio = 0.0 if step_max <= 0 else float(step_now) / float(step_max)
                                        progress_bar.progress(min((job_index + ratio) / float(total_steps), 1.0))
                                        if status_text and (_seed is not None) and (_mode is not None):
                                            status_text.text(get_text("ablation_running", lang, _mode, _seed) + f" ({int(step_now)}/{int(step_max)})")
                                    except Exception:
                                        pass
                                return _cb

                            for s in seeds_used:
                                # 禁用动态事件
                                status_text.text(get_text("ablation_running", lang, "OFF", s) + f" (0/{int(ablation_max_steps)})")
                                d_stats, d_hist, d_score, d_reward = run_scheduling(
                                    actor_model,
                                    orders_cfg,
                                    custom_products,
                                    max_steps=int(ablation_max_steps),
                                    enable_failure=False,
                                    enable_emergency=False,
                                    failure_config=None,
                                    emergency_config=None,
                                    progress_bar=None,
                                    status_text=None,
                                    seed=int(s),
                                    progress_callback=(lambda step_now, step_max, _cb=_make_step_cb(done), _s=int(s): _cb(step_now, step_max, _seed=_s, _mode="OFF")),
                                    progress_callback_interval=10
                                )
                                disabled_runs.append({
                                    'stats': d_stats,
                                    'score': d_score,
                                    'total_reward': d_reward,
                                    'gantt_history': d_hist,
                                    'seed': int(s)
                                })
                                done += 1
                                progress_bar.progress(done / total_steps)

                                if include_heuristics_ablation:
                                    for h in ['FIFO', 'EDD', 'SPT', 'ATC']:
                                        h_stats, h_hist, h_score = run_heuristic_scheduling(
                                            h,
                                            orders_cfg,
                                            custom_products,
                                            max_steps=int(ablation_max_steps),
                                            progress_bar=None,
                                            status_text=None,
                                            enable_failure=False,
                                            enable_emergency=False,
                                            failure_config=None,
                                            emergency_config=None,
                                            seed=int(s),
                                            progress_callback=(lambda step_now, step_max, _cb=_make_step_cb(done), _s=int(s): _cb(step_now, step_max, _seed=_s, _mode="OFF")),
                                            progress_callback_interval=10
                                        )
                                        heuristic_disabled[h].append({
                                            'stats': h_stats,
                                            'score': h_score,
                                            'gantt_history': h_hist,
                                            'seed': int(s)
                                        })
                                        done += 1
                                        progress_bar.progress(done / total_steps)

                                # 启用动态事件（使用当前页面设置）
                                status_text.text(get_text("ablation_running", lang, "ON", s) + f" (0/{int(ablation_max_steps)})")
                                e_stats, e_hist, e_score, e_reward = run_scheduling(
                                    actor_model,
                                    orders_cfg,
                                    custom_products,
                                    max_steps=int(ablation_max_steps),
                                    enable_failure=enabled_failure,
                                    enable_emergency=enabled_emergency,
                                    failure_config=enabled_failure_cfg,
                                    emergency_config=enabled_emergency_cfg,
                                    progress_bar=None,
                                    status_text=None,
                                    seed=int(s),
                                    progress_callback=(lambda step_now, step_max, _cb=_make_step_cb(done), _s=int(s): _cb(step_now, step_max, _seed=_s, _mode="ON")),
                                    progress_callback_interval=10
                                )
                                enabled_runs.append({
                                    'stats': e_stats,
                                    'score': e_score,
                                    'total_reward': e_reward,
                                    'gantt_history': e_hist,
                                    'seed': int(s)
                                })
                                done += 1
                                progress_bar.progress(done / total_steps)

                                if include_heuristics_ablation:
                                    for h in ['FIFO', 'EDD', 'SPT', 'ATC']:
                                        h_stats, h_hist, h_score = run_heuristic_scheduling(
                                            h,
                                            orders_cfg,
                                            custom_products,
                                            max_steps=int(ablation_max_steps),
                                            progress_bar=None,
                                            status_text=None,
                                            enable_failure=enabled_failure,
                                            enable_emergency=enabled_emergency,
                                            failure_config=enabled_failure_cfg,
                                            emergency_config=enabled_emergency_cfg,
                                            seed=int(s),
                                            progress_callback=(lambda step_now, step_max, _cb=_make_step_cb(done), _s=int(s): _cb(step_now, step_max, _seed=_s, _mode="ON")),
                                            progress_callback_interval=10
                                        )
                                        heuristic_enabled[h].append({
                                            'stats': h_stats,
                                            'score': h_score,
                                            'gantt_history': h_hist,
                                            'seed': int(s)
                                        })
                                        done += 1
                                        progress_bar.progress(done / total_steps)

                            progress_bar.empty()
                            status_text.empty()

                            st.session_state['dynamic_event_ablation_results'] = {
                                'model_name': selected_ablation_model,
                                'model_path': model_path,
                                'seeds_used': seeds_used,
                                'disabled': disabled_runs,
                                'enabled': enabled_runs,
                                'heuristic_disabled': heuristic_disabled,
                                'heuristic_enabled': heuristic_enabled,
                            }

                            # 落盘保存（JSON不含gantt_history，甘特图另存HTML）
                            try:
                                save_dynamic_event_ablation_results(st.session_state['dynamic_event_ablation_results'])
                            except Exception:
                                pass
                            st.success(get_text("ablation_completed", lang))
                            st.rerun()
                    except Exception as e:
                        st.error(f"{get_text('ablation_failed', lang)}{str(e)}")

                if 'dynamic_event_ablation_results' in st.session_state and st.session_state['dynamic_event_ablation_results']:
                    res = st.session_state['dynamic_event_ablation_results']
                    if res.get('model_name') == selected_ablation_model:
                        def _avg(lst, key_fn):
                            return sum(key_fn(x) for x in lst) / max(1, len(lst))

                        def _std(lst, key_fn):
                            vals = [float(key_fn(x)) for x in (lst or [])]
                            if len(vals) <= 1:
                                return 0.0
                            return float(np.std(vals, ddof=1))

                        def _get_stat_int(run: dict, key: str) -> int:
                            try:
                                return int((run or {}).get('stats', {}).get(key, 0) or 0)
                            except Exception:
                                return 0

                        def _get_stat_float(run: dict, key: str) -> float:
                            try:
                                return float((run or {}).get('stats', {}).get(key, 0.0) or 0.0)
                            except Exception:
                                return 0.0

                        disabled_runs = res.get('disabled', [])
                        enabled_runs = res.get('enabled', [])
                        if disabled_runs and enabled_runs:
                            total_parts_target = sum(order['quantity'] for order in st.session_state['orders'])
                            table = []
                            def _append_row(alg_name: str, tag: str, runs: list):
                                avg_parts = _avg(runs, lambda r: r['stats'].get('total_parts', 0))
                                avg_makespan = _avg(runs, lambda r: r['stats'].get('makespan', 0.0))
                                avg_util = _avg(runs, lambda r: r['stats'].get('mean_utilization', 0.0))
                                avg_tard = _avg(runs, lambda r: r['stats'].get('total_tardiness', 0.0))
                                avg_score = _avg(runs, lambda r: r.get('score', 0.0))
                                std_score = _std(runs, lambda r: r.get('score', 0.0))
                                avg_failures = _avg(runs, lambda r: _get_stat_int(r, 'equipment_failure_event_count'))
                                avg_emerg_orders = _avg(runs, lambda r: _get_stat_int(r, 'emergency_orders_inserted_count'))
                                avg_emerg_parts = _avg(runs, lambda r: _get_stat_int(r, 'emergency_parts_inserted_count'))
                                completion_rate_pct = (avg_parts / total_parts_target * 100) if total_parts_target > 0 else 0
                                table.append({
                                    get_text("algorithm", lang): alg_name,
                                    get_text("ablation_group", lang): tag,
                                    get_text("completion_rate", lang): f"{completion_rate_pct:.1f}%",
                                    get_text("avg_makespan", lang): f"{avg_makespan:.1f}",
                                    get_text("avg_utilization", lang): f"{avg_util*100:.1f}%",
                                    get_text("avg_tardiness", lang): f"{avg_tard:.1f}",
                                    get_text("avg_score", lang): f"{avg_score:.3f}",
                                    "score_std": f"{std_score:.3f}",
                                    "故障次数(均值)": f"{avg_failures:.2f}",
                                    "插单数(均值)": f"{avg_emerg_orders:.2f}",
                                    "插入零件(均值)": f"{avg_emerg_parts:.2f}",
                                    get_text("runs", lang): len(runs),
                                })

                            _append_row("MARL (PPO)", "OFF", disabled_runs)
                            _append_row("MARL (PPO)", "ON", enabled_runs)

                            h_dis = res.get('heuristic_disabled', {}) or {}
                            h_en = res.get('heuristic_enabled', {}) or {}
                            for h in ['FIFO', 'EDD', 'SPT', 'ATC']:
                                if h in h_dis and h_dis[h]:
                                    _append_row(h, "OFF", h_dis[h])
                                if h in h_en and h_en[h]:
                                    _append_row(h, "ON", h_en[h])

                            df_table = pd.DataFrame(table)
                            st.dataframe(df_table, use_container_width=True, hide_index=True)

                            def _collect_score_points(alg_name: str, tag: str, runs: list):
                                rows = []
                                for r in (runs or []):
                                    try:
                                        rows.append({
                                            'algorithm': alg_name,
                                            'group': tag,
                                            'seed': r.get('seed', None),
                                            'score': float(r.get('score', 0.0) or 0.0),
                                            'tardiness': _get_stat_float(r, 'total_tardiness'),
                                            'makespan': _get_stat_float(r, 'makespan'),
                                            'failures': _get_stat_int(r, 'equipment_failure_event_count'),
                                            'emergency_orders': _get_stat_int(r, 'emergency_orders_inserted_count'),
                                        })
                                    except Exception:
                                        pass
                                return rows

                            points = []
                            points += _collect_score_points('MARL (PPO)', 'OFF', disabled_runs)
                            points += _collect_score_points('MARL (PPO)', 'ON', enabled_runs)
                            for h in ['FIFO', 'EDD', 'SPT', 'ATC']:
                                if h in h_dis and h_dis[h]:
                                    points += _collect_score_points(h, 'OFF', h_dis[h])
                                if h in h_en and h_en[h]:
                                    points += _collect_score_points(h, 'ON', h_en[h])
                            df_points = pd.DataFrame(points)

                            if not df_points.empty:
                                st.subheader("动态事件鲁棒性摘要")

                                def _mean_of(alg: str, group: str) -> float:
                                    try:
                                        sub = df_points[(df_points['algorithm'] == alg) & (df_points['group'] == group)]
                                        if sub.empty:
                                            return 0.0
                                        return float(sub['score'].mean())
                                    except Exception:
                                        return 0.0

                                def _std_of(alg: str, group: str) -> float:
                                    try:
                                        sub = df_points[(df_points['algorithm'] == alg) & (df_points['group'] == group)]
                                        if len(sub) <= 1:
                                            return 0.0
                                        return float(sub['score'].std(ddof=1))
                                    except Exception:
                                        return 0.0

                                marl_off = _mean_of('MARL (PPO)', 'OFF')
                                marl_on = _mean_of('MARL (PPO)', 'ON')
                                marl_drop = (marl_on - marl_off)
                                marl_drop_pct = (marl_drop / marl_off * 100.0) if marl_off > 1e-9 else 0.0

                                best_heur_on = None
                                best_heur_name = None
                                for h in ['FIFO', 'EDD', 'SPT', 'ATC']:
                                    v = _mean_of(h, 'ON')
                                    if best_heur_on is None or v > best_heur_on:
                                        best_heur_on = v
                                        best_heur_name = h

                                marl_adv = (marl_on - (best_heur_on if best_heur_on is not None else 0.0))
                                marl_adv_pct = (marl_adv / best_heur_on * 100.0) if (best_heur_on is not None and best_heur_on > 1e-9) else 0.0

                                c1, c2, c3, c4 = st.columns(4)
                                c1.metric("MARL 平均分 (OFF)", f"{marl_off:.3f}")
                                c2.metric("MARL 平均分 (ON)", f"{marl_on:.3f}", f"{marl_drop:+.3f} ({marl_drop_pct:+.1f}%)")
                                c3.metric("MARL 稳定性(ON std)", f"{_std_of('MARL (PPO)', 'ON'):.3f}")
                                label_best = f"相对最强启发式(ON): {best_heur_name}" if best_heur_name else "相对启发式(ON)"
                                c4.metric(label_best, f"{marl_adv:+.3f}", f"{marl_adv_pct:+.1f}%")

                                st.subheader("分数分布（箱线图）")
                                fig = go.Figure()
                                for alg in df_points['algorithm'].unique():
                                    for grp in ['OFF', 'ON']:
                                        sub = df_points[(df_points['algorithm'] == alg) & (df_points['group'] == grp)]
                                        if sub.empty:
                                            continue
                                        name = f"{alg} / {grp}"
                                        fig.add_trace(go.Box(y=sub['score'], name=name, boxmean='sd'))
                                fig.update_layout(
                                    height=420,
                                    margin=dict(l=10, r=10, t=10, b=10),
                                    yaxis_title="score"
                                )
                                st.plotly_chart(fig, use_container_width=True)

                                st.subheader("按 seed 配对的提升（Pairwise Gains）")

                                present_heur = [h for h in ['FIFO', 'EDD', 'SPT', 'ATC'] if h in df_points['algorithm'].unique()]
                                baseline_options = ['最强启发式(ON)'] + present_heur
                                baseline_choice = st.selectbox(
                                    "基线(ON)",
                                    options=baseline_options,
                                    index=0,
                                    key="ablation_pairwise_baseline"
                                )

                                # MARL ON vs MARL OFF（同seed差值）
                                try:
                                    marl_on_df = df_points[(df_points['algorithm'] == 'MARL (PPO)') & (df_points['group'] == 'ON')][['seed', 'score']].rename(columns={'score': 'marl_on'})
                                    marl_off_df = df_points[(df_points['algorithm'] == 'MARL (PPO)') & (df_points['group'] == 'OFF')][['seed', 'score']].rename(columns={'score': 'marl_off'})
                                    merged_marl = pd.merge(marl_on_df, marl_off_df, on='seed', how='inner')
                                    merged_marl['delta_on_off'] = merged_marl['marl_on'] - merged_marl['marl_off']
                                except Exception:
                                    merged_marl = pd.DataFrame()

                                if not merged_marl.empty:
                                    merged_marl = merged_marl.sort_values('seed')
                                    fig_pair1 = go.Figure()
                                    fig_pair1.add_trace(go.Bar(
                                        x=merged_marl['seed'].astype(str),
                                        y=merged_marl['delta_on_off'],
                                        name='MARL(ON) - MARL(OFF)'
                                    ))
                                    fig_pair1.add_shape(
                                        type='line',
                                        xref='paper',
                                        yref='y',
                                        x0=0.0,
                                        x1=1.0,
                                        y0=0.0,
                                        y1=0.0,
                                        line=dict(color='rgba(0,0,0,0.35)', width=1)
                                    )
                                    fig_pair1.update_layout(
                                        height=360,
                                        margin=dict(l=10, r=10, t=10, b=10),
                                        yaxis_title='Δscore'
                                    )
                                    st.plotly_chart(fig_pair1, use_container_width=True)

                                # MARL ON vs baseline ON（同seed差值）
                                try:
                                    if baseline_choice == '最强启发式(ON)':
                                        baseline_alg = best_heur_name
                                    else:
                                        baseline_alg = baseline_choice

                                    base_on_df = df_points[(df_points['algorithm'] == baseline_alg) & (df_points['group'] == 'ON')][['seed', 'score']].rename(columns={'score': 'baseline_on'})
                                    marl_on_df2 = df_points[(df_points['algorithm'] == 'MARL (PPO)') & (df_points['group'] == 'ON')][['seed', 'score']].rename(columns={'score': 'marl_on'})
                                    merged_vs = pd.merge(marl_on_df2, base_on_df, on='seed', how='inner')
                                    merged_vs['delta_vs_baseline'] = merged_vs['marl_on'] - merged_vs['baseline_on']
                                except Exception:
                                    merged_vs = pd.DataFrame()

                                if not merged_vs.empty:
                                    merged_vs = merged_vs.sort_values('seed')
                                    fig_pair2 = go.Figure()
                                    fig_pair2.add_trace(go.Bar(
                                        x=merged_vs['seed'].astype(str),
                                        y=merged_vs['delta_vs_baseline'],
                                        name=f"MARL(ON) - {baseline_alg}(ON)"
                                    ))
                                    fig_pair2.add_shape(
                                        type='line',
                                        xref='paper',
                                        yref='y',
                                        x0=0.0,
                                        x1=1.0,
                                        y0=0.0,
                                        y1=0.0,
                                        line=dict(color='rgba(0,0,0,0.35)', width=1)
                                    )
                                    fig_pair2.update_layout(
                                        height=360,
                                        margin=dict(l=10, r=10, t=10, b=10),
                                        yaxis_title='Δscore'
                                    )
                                    st.plotly_chart(fig_pair2, use_container_width=True)

    
    with st.expander(get_text("model_comparison_description", lang), expanded=False):
        st.markdown(get_text("model_comparison_help", lang))
        
        # 检查是否满足对比条件
        if not st.session_state.get('orders', []):
            st.warning(get_text("config_orders_first_comparison", lang))
        else:
            st.info(f"📋 {get_text('current_orders_count', lang, len(st.session_state['orders']))}")
            
            # 显示当前动态环境配置
            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"⚙️ {get_text('equipment_failure', lang)}: {'✅ ' + get_text('enabled', lang) if st.session_state.get('enable_failure', False) else '❌ ' + get_text('disabled', lang)}")
            with col2:
                st.caption(f"📥 {get_text('emergency_orders', lang)}: {'✅ ' + get_text('enabled', lang) if st.session_state.get('enable_emergency', False) else '❌ ' + get_text('disabled', lang)}")
            
            st.divider()
            
            # 模型选择区域
            st.subheader(get_text("select_models_to_compare", lang))
            
            available_models = find_available_models()
            if not available_models:
                st.warning(get_text("no_model_found", lang))
            else:
                # 使用下拉多选框选择模型
                model_options = {m["name"]: m["path"] for m in available_models}
                
                # 初始化默认选择
                default_selection = list(model_options.keys())[:min(3, len(model_options))]
                
                selected_model_names = st.multiselect(
                    get_text("select_models", lang),
                    options=list(model_options.keys()),
                    default=default_selection,
                    help=get_text("select_models_help", lang),
                    key="model_comparison_multiselect"
                )
                
                st.divider()
                
                if len(selected_model_names) < 2:
                    st.warning(get_text("select_at_least_two_models", lang))
                else:
                    # 显示已选模型列表
                    st.success(f"✅ {get_text('selected_models_count', lang, len(selected_model_names))}")
                    
                    # 清晰展示已选模型
                    st.caption(get_text("selected_models_list", lang))
                    for idx, model_name in enumerate(selected_model_names, 1):
                        st.markdown(f"**{idx}.** `{model_name}`")
                    
                    # 对比配置选项
                    st.subheader(get_text("comparison_parameters", lang))
                    col1, col2 = st.columns(2)
                    with col1:
                        max_steps_comparison = int(st.session_state.get('max_steps_single', 1500))
                        st.caption(f"{get_text('max_steps', lang)}: {int(max_steps_comparison)}")
                    with col2:
                        seed_mode = st.toggle(
                            get_text("seed_mode", lang),
                            value=False,
                            help=get_text("seed_mode_help", lang),
                            key="comparison_multi_seed"
                        )

                        comparison_runs = st.number_input(
                            get_text("comparison_runs", lang),
                            min_value=1,
                            max_value=5,
                            value=2,
                            step=1,
                            disabled=(not seed_mode),
                            help=get_text("comparison_runs_help", lang),
                            key="comparison_runs"
                        )

                    base_seed = st.number_input(
                        get_text("base_seed", lang),
                        min_value=0,
                        max_value=1000000,
                        value=42,
                        step=1,
                        help=get_text("base_seed_help", lang),
                        key="comparison_base_seed"
                    )

                    include_heuristics_comparison = st.checkbox(
                        get_text("include_heuristics_baseline", lang),
                        value=bool(st.session_state.get('include_heuristics_comparison', True)),
                        help=get_text("include_heuristics_baseline_help", lang),
                        key="include_heuristics_comparison"
                    )
                    
                    # 开始对比按钮
                    if st.button(get_text("start_comparison", lang), type="primary", use_container_width=True):
                        # 初始化对比结果存储
                        comparison_results = {}

                        heuristic_baselines = {h: [] for h in ['FIFO', 'EDD', 'SPT', 'ATC']} if include_heuristics_comparison else {}

                        effective_runs = int(comparison_runs) if seed_mode else 1
                        seeds_used = [int(base_seed + i) for i in range(effective_runs)] if seed_mode else [int(base_seed)]
                        
                        # 创建进度条
                        total_runs = len(selected_model_names) * effective_runs
                        if include_heuristics_comparison:
                            total_runs += 4 * effective_runs
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        job_done = 0
                        
                        # 依次运行每个模型
                        for model_name in selected_model_names:
                            model_path = model_options[model_name]
                            model_results = []
                            
                            for run_idx in range(effective_runs):
                                status_text.text(f"{get_text('running_model', lang, model_name, run_idx + 1, effective_runs)}")
                                progress_bar.progress(min(job_done / total_runs, 1.0))

                                current_seed = int(seeds_used[min(run_idx, len(seeds_used) - 1)])
                                
                                # 加载模型
                                actor_model, _, message = load_model(model_path)
                                if actor_model is None:
                                    st.error(f"{get_text('load_model_failed', lang, model_name)}: {message}")
                                    continue
                                
                                # 运行调度
                                try:
                                    def _step_cb(step_now: int, step_max: int, _base=job_done, _tot=total_runs):
                                        try:
                                            ratio = 0.0 if step_max <= 0 else float(step_now) / float(step_max)
                                            progress_bar.progress(min((_base + ratio) / float(_tot), 1.0))
                                            status_text.text(f"{get_text('running_model', lang, model_name, run_idx + 1, effective_runs)} ({int(step_now)}/{int(step_max)})")
                                        except Exception:
                                            pass

                                    final_stats, gantt_history, score, total_reward = run_scheduling(
                                        actor_model,
                                        st.session_state['orders'],
                                        st.session_state.get('custom_products'),
                                        max_steps=max_steps_comparison,
                                        enable_failure=st.session_state.get('enable_failure', False),
                                        enable_emergency=st.session_state.get('enable_emergency', False),
                                        failure_config=st.session_state.get('failure_config', {}),
                                        emergency_config=st.session_state.get('emergency_config', {}),
                                        progress_bar=None,  # 不显示子进度条
                                        status_text=None,
                                        seed=current_seed,
                                        progress_callback=_step_cb,
                                        progress_callback_interval=10
                                    )
                                    
                                    model_results.append({
                                        'stats': final_stats,
                                        'score': score,
                                        'total_reward': total_reward,
                                        'gantt_history': gantt_history,
                                        'seed': current_seed
                                    })
                                    job_done += 1
                                except Exception as e:
                                    st.error(f"{get_text('scheduling_failed', lang, model_name)}: {str(e)}")
                                    continue
                            
                            # 保存该模型的所有运行结果
                            if model_results:
                                comparison_results[model_name] = model_results

                        if include_heuristics_comparison:
                            for h in ['FIFO', 'EDD', 'SPT', 'ATC']:
                                for run_idx in range(effective_runs):
                                    current_seed = int(seeds_used[min(run_idx, len(seeds_used) - 1)])
                                    try:
                                        status_text.text(get_text('running_model', lang, h, run_idx + 1, effective_runs))
                                        progress_bar.progress(min(job_done / total_runs, 1.0))

                                        def _h_step_cb(step_now: int, step_max: int, _base=job_done, _tot=total_runs):
                                            try:
                                                ratio = 0.0 if step_max <= 0 else float(step_now) / float(step_max)
                                                progress_bar.progress(min((_base + ratio) / float(_tot), 1.0))
                                                status_text.text(get_text('running_model', lang, h, run_idx + 1, effective_runs) + f" ({int(step_now)}/{int(step_max)})")
                                            except Exception:
                                                pass

                                        h_stats, h_history, h_score = run_heuristic_scheduling(
                                            h,
                                            st.session_state['orders'],
                                            st.session_state.get('custom_products'),
                                            max_steps=max_steps_comparison,
                                            progress_bar=None,
                                            status_text=None,
                                            enable_failure=st.session_state.get('enable_failure', False),
                                            enable_emergency=st.session_state.get('enable_emergency', False),
                                            failure_config=st.session_state.get('failure_config', {}),
                                            emergency_config=st.session_state.get('emergency_config', {}),
                                            seed=current_seed,
                                            progress_callback=_h_step_cb,
                                            progress_callback_interval=10
                                        )
                                        heuristic_baselines[h].append({
                                            'stats': h_stats,
                                            'score': h_score,
                                            'total_reward': 0.0,
                                            'gantt_history': h_history,
                                            'seed': current_seed
                                        })
                                        job_done += 1
                                    except Exception:
                                        continue
                        
                        progress_bar.empty()
                        status_text.empty()
                        
                        # 保存对比结果到session_state
                        if comparison_results:
                            st.session_state['model_comparison_results'] = comparison_results
                            if include_heuristics_comparison:
                                st.session_state['model_comparison_heuristics'] = heuristic_baselines
                            
                            # 保存模型对比结果到文件
                            save_model_comparison_results(comparison_results, selected_model_names, seeds_used=seeds_used, heuristic_baselines=heuristic_baselines)
                            
                            st.success(get_text("comparison_completed", lang))
                            st.rerun()
                        else:
                            st.error(get_text("comparison_failed", lang))
            
            # 显示对比结果
            if 'model_comparison_results' in st.session_state and st.session_state['model_comparison_results']:
                st.divider()
                st.subheader(get_text("comparison_results", lang))
                
                results = st.session_state['model_comparison_results']
                heuristics = st.session_state.get('model_comparison_heuristics', {})
                if heuristics:
                    for h, runs in heuristics.items():
                        if runs:
                            results = {**results, **{h: runs}}
                
                # 准备对比数据
                comparison_data = []
                for model_name, runs in results.items():
                    # 计算平均值
                    avg_completion_rate = sum(r['stats']['total_parts'] for r in runs) / len(runs)
                    avg_makespan = sum(r['stats']['makespan'] for r in runs) / len(runs)
                    avg_utilization = sum(r['stats']['mean_utilization'] for r in runs) / len(runs)
                    avg_tardiness = sum(r['stats']['total_tardiness'] for r in runs) / len(runs)
                    avg_score = sum(r['score'] for r in runs) / len(runs)
                    avg_reward = sum(r['total_reward'] for r in runs) / len(runs)
                    
                    # 计算目标完工件数
                    total_parts_target = sum(order['quantity'] for order in st.session_state['orders'])
                    completion_rate_pct = (avg_completion_rate / total_parts_target * 100) if total_parts_target > 0 else 0
                    
                    comparison_data.append({
                        get_text("model_name", lang): model_name,
                        get_text("completion_rate", lang): f"{completion_rate_pct:.1f}%",
                        get_text("avg_makespan", lang): f"{avg_makespan:.1f}",
                        get_text("avg_utilization", lang): f"{avg_utilization*100:.1f}%",
                        get_text("avg_tardiness", lang): f"{avg_tardiness:.1f}",
                        get_text("avg_score", lang): f"{avg_score:.3f}",
                        get_text("avg_reward", lang): f"{avg_reward:.1f}",
                        get_text("runs", lang): len(runs)
                    })
                
                # 显示对比表格
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                
                # 显示雷达图对比
                st.subheader(get_text("radar_chart_comparison", lang))
                
                # 准备雷达图数据（归一化到0-1）
                metrics = ['completion_rate', 'utilization', 'score']
                fig = go.Figure()
                
                for model_name, runs in results.items():
                    avg_completion_rate = sum(r['stats']['total_parts'] for r in runs) / len(runs)
                    avg_utilization = sum(r['stats']['mean_utilization'] for r in runs) / len(runs)
                    avg_score = sum(r['score'] for r in runs) / len(runs)
                    
                    total_parts_target = sum(order['quantity'] for order in st.session_state['orders'])
                    completion_rate_norm = (avg_completion_rate / total_parts_target) if total_parts_target > 0 else 0
                    
                    fig.add_trace(go.Scatterpolar(
                        r=[completion_rate_norm, avg_utilization, avg_score],
                        theta=[get_text("completion_rate", lang), get_text("utilization", lang), get_text("score", lang)],
                        fill='toself',
                        name=model_name
                    ))
                
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=True,
                    title=get_text("model_performance_radar", lang)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 显示柱状图对比
                st.subheader(get_text("bar_chart_comparison", lang))
                
                # 准备柱状图数据
                fig_bar = go.Figure()
                model_names = list(results.keys())
                
                # 完工率
                completion_rates = []
                for model_name in model_names:
                    runs = results[model_name]
                    avg_completion = sum(r['stats']['total_parts'] for r in runs) / len(runs)
                    total_target = sum(order['quantity'] for order in st.session_state['orders'])
                    completion_rates.append((avg_completion / total_target * 100) if total_target > 0 else 0)
                
                fig_bar.add_trace(go.Bar(
                    name=get_text("completion_rate", lang),
                    x=model_names,
                    y=completion_rates,
                    text=[f"{v:.1f}%" for v in completion_rates],
                    textposition='auto'
                ))
                
                fig_bar.update_layout(
                    title=get_text("completion_rate_comparison", lang),
                    xaxis_title=get_text("model_name", lang),
                    yaxis_title=get_text("completion_rate", lang) + " (%)",
                    showlegend=False
                )
                
                st.plotly_chart(fig_bar, use_container_width=True)
                
                # 清除对比结果按钮
                if st.button(get_text("clear_comparison_results", lang)):
                    del st.session_state['model_comparison_results']
                    st.rerun()

if __name__ == "__main__":
    setup_page()
    main()
