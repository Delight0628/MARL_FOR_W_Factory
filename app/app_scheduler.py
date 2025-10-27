"""
W工厂智能调度应用 - 基于MARL的生产调度系统
支持模型加载、订单配置和调度结果可视化
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
from datetime import datetime
import json
from i18n import LANGUAGES, get_text
import gymnasium as gym  # 10-25-14-30 引入以识别MultiDiscrete动作空间

if os.environ.get('FORCE_CPU', '0') == '1':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽TensorFlow的INFO级别日志

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
                from mappo.ppo_marl_train import PPONetwork
                
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
    config_file = os.path.join(app_dir, "custom_products.json")
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_custom_products(products):
    """保存自定义产品配置到文件"""
    config_file = os.path.join(app_dir, "custom_products.json")
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(products, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        st.error(get_text("save_failed", get_language(), str(e)))
        return False

def load_app_state():
    """从文件加载应用状态（订单配置、模型路径、仿真结果等）"""
    state_file = os.path.join(app_dir, "app_state.json")
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

def save_app_state():
    """保存应用状态到文件"""
    state_file = os.path.join(app_dir, "app_state.json")
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
                'total_reward': st.session_state.get('last_total_reward', None)
            },
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state_to_save, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        # 静默失败，不显示错误
        return False

def calculate_product_total_time(product: str, product_routes: dict) -> float:
    """计算产品总加工时间"""
    route = product_routes.get(product, [])
    return sum(step["time"] for step in route)

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

@st.cache_resource
def load_model(model_path):
    """加载训练好的模型"""
    try:
        # 10-26-16-00 使用健壮的加载函数
        actor_model = load_actor_model_robust(model_path)
        if actor_model is None:
            return None, get_text("error_load_model_failed", get_language(), "所有加载策略均失败")
        
        return actor_model, get_text("model_loaded_successfully", get_language())
    except Exception as e:
        return None, get_text("error_load_model_failed", get_language(), str(e))

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
                    for file in os.listdir(run_path):
                        if file.endswith("_actor.keras"):
                            model_path = os.path.join(run_path, file)
                            model_name = file.replace("_actor.keras", "")
                            models.append({
                                # 使用 "实验目录/模型名" 的格式，更具描述性
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

def run_scheduling(actor_model, orders_config, custom_products=None, max_steps=1500, progress_bar=None, status_text=None):
    """运行调度仿真"""
    # 如果有自定义产品，临时添加到PRODUCT_ROUTES
    from environments import w_factory_config
    original_routes = None
    
    if custom_products:
        original_routes = w_factory_config.PRODUCT_ROUTES.copy()
        w_factory_config.PRODUCT_ROUTES.update(custom_products)
    
    try:
        config = {
            'custom_orders': orders_config,
            'disable_failures': True,
            'stage_name': '用户自定义调度'
        }
        
        if status_text:
            status_text.text(get_text("initializing", get_language()))
        
        env = WFactoryEnv(config=config)
        obs, info = env.reset(seed=42)
        
        step_count = 0
        total_reward = 0
        
        if status_text:
            status_text.text(get_text("starting_sim", get_language()))
        
        while step_count < max_steps:
            actions = {}
            for agent in env.agents:
                if agent in obs:
                    state = tf.expand_dims(obs[agent], 0)
                    # 10-25-14-30 统一：兼容多头/单头输出并采用按头无放回贪心选择
                    action_probs_tensor = actor_model(state, training=False)
                    if isinstance(action_probs_tensor, (list, tuple)):
                        head_probs_list = [np.squeeze(h.numpy()) for h in action_probs_tensor]
                    else:
                        head_probs_list = [np.squeeze(action_probs_tensor.numpy()[0])]
                    sp = env.action_space(agent)
                    if isinstance(sp, gym.spaces.MultiDiscrete):
                        k = len(sp.nvec)
                        chosen = []
                        used = set()
                        for i in range(k):
                            base = head_probs_list[i] if i < len(head_probs_list) else head_probs_list[0]
                            p = np.asarray(base, dtype=np.float64)
                            p = np.clip(p, 1e-12, np.inf)
                            if used:
                                idxs = list(used)
                                p[idxs] = 0.0
                            s = p.sum()
                            if s <= 1e-12:
                                idx = 0
                            else:
                                p = p / s
                                idx = int(np.argmax(p))
                            chosen.append(idx)
                            used.add(idx)
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
            
            if any(terminations.values()) or any(truncations.values()):
                break
        
        if status_text:
            status_text.text(get_text("generating_results", get_language()))
        
        if progress_bar:
            progress_bar.progress(1.0)
        
        final_stats = env.sim.get_final_stats()
        gantt_history = env.sim.gantt_chart_history
        score = calculate_episode_score(final_stats, config)
        
        env.close()
        
        if status_text:
            status_text.text(get_text("scheduling_complete", get_language()))
        
        return final_stats, gantt_history, score, total_reward
    finally:
        # 恢复原始产品路线
        if original_routes is not None:
            w_factory_config.PRODUCT_ROUTES = original_routes

def create_gantt_chart(history):
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
                state_file = os.path.join(app_dir, "app_state.json")
                custom_file = os.path.join(app_dir, "custom_products.json")
                
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
            
            # 恢复仿真结果
            last_sim = saved_state.get('last_simulation', {})
            if last_sim.get('stats'):
                st.session_state['last_stats'] = last_sim.get('stats')
                st.session_state['last_gantt_history'] = last_sim.get('gantt_history')
                st.session_state['last_score'] = last_sim.get('score')
                st.session_state['last_total_reward'] = last_sim.get('total_reward')
                
                # 同时设置到当前结果变量中，以便显示
                st.session_state['final_stats'] = last_sim.get('stats')
                st.session_state['gantt_history'] = last_sim.get('gantt_history')
                st.session_state['score'] = last_sim.get('score')
                st.session_state['total_reward'] = last_sim.get('total_reward')
                st.session_state['show_results'] = True
                
            # 如果有保存的模型路径，尝试重新加载模型
            if saved_state.get('model_loaded') and saved_state.get('model_path'):
                try:
                    model, msg = load_model(saved_state['model_path'])
                    if model is not None:
                        st.session_state['actor_model'] = model
                except:
                    pass
        
        st.session_state['state_loaded'] = True
    
    # 步骤1：模型加载
    st.header(get_text("system_config", lang))
    
    # 模型加载方式选择
    model_input_method = st.radio(
        get_text("model_loading_method", lang),
        [get_text("from_history", lang), get_text("manual_input", lang)],
        horizontal=True
    )
    
    actor_model = None
    
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
                    get_text("select_model", lang),
                    options=model_options,
                    index=default_index,
                    help=get_text("model_help", lang)
                )
                
                selected_model_info = next(m for m in available_models if m["name"] == selected_model)
                model_path = selected_model_info["path"]
                
                st.caption(f"{get_text('model_path', lang)}{model_path}")
        else:
            model_path = st.text_input(
                get_text("model_path_input", lang),
                value="mappo/ppo_models/",
                help=get_text("model_path_help", lang)
            )
    
    with col2:
        st.write("")  # 空行对齐
        st.write("")  # 空行对齐
        # 加载模型按钮
        if st.button(get_text("load_model", lang), type="primary", use_container_width=True):
            if model_path:
                with st.spinner(get_text("loading_model", lang)):
                    actor_model, message = load_model(model_path)
                    if actor_model is not None:
                        st.session_state['actor_model'] = actor_model
                        st.session_state['model_path'] = model_path
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
                col1, col2 = st.columns([0, 1])
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
    config_method = st.radio(
        get_text("choose_config_method", lang),
        [get_text("random_orders", lang), get_text("custom_orders", lang)],
        horizontal=True,
        label_visibility="collapsed"
    )
    
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
            
            submitted = st.form_submit_button(get_text("add_order_button", lang))
            if submitted:
                order = {
                    "product": product,
                    "quantity": int(quantity),
                    "priority": int(priority),
                    "arrival_time": int(arrival_time),
                    "due_date": int(due_date)
                }
                st.session_state['orders'].append(order)
                save_app_state()  # 💾 保存状态
                st.success(get_text("order_added_full", lang, product, quantity, arrival_time, due_date))
                st.rerun()
    
    else:  # 随机生成订单
        st.subheader(get_text("random_order_gen", lang))
        
        # 订单数量
        num_orders = st.slider(get_text("order_count", lang), min_value=3, max_value=10, value=5)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write(get_text("product_quantity_range", lang))
            subcol1, subcol2 = st.columns(2)
            with subcol1:
                min_quantity = st.number_input(get_text("from", lang), min_value=1, max_value=50, value=3, key="qty_min")
            with subcol2:
                max_quantity = st.number_input(get_text("to", lang), min_value=1, max_value=50, value=10, key="qty_max")
        
        with col2:
            st.write(get_text("due_date_range", lang))
            subcol1, subcol2 = st.columns(2)
            with subcol1:
                min_due = st.number_input(get_text("from", lang), min_value=100, max_value=2000, value=200, step=10, key="due_min")
            with subcol2:
                max_due = st.number_input(get_text("to", lang), min_value=100, max_value=2000, value=700, step=10, key="due_max")
        
        with col3:
            st.write(get_text("arrival_time_range", lang))
            subcol1, subcol2 = st.columns(2)
            with subcol1:
                min_arrival = st.number_input(get_text("from", lang), min_value=0, max_value=500, value=0, step=10, key="arrival_min")
            with subcol2:
                max_arrival = st.number_input(get_text("to", lang), min_value=0, max_value=500, value=50, step=10, key="arrival_max")
        
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
        if st.button(get_text("start_simulation", lang), type="primary", use_container_width=True):
            try:
                actor_model = st.session_state['actor_model']
                orders = st.session_state['orders']
                custom_products = st.session_state.get('custom_products', {})
                
                # 创建进度条和状态文本
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                final_stats, gantt_history, score, total_reward = run_scheduling(
                    actor_model, orders, custom_products, 
                    progress_bar=progress_bar, 
                    status_text=status_text
                )
                
                # 保存结果到session state
                st.session_state['final_stats'] = final_stats
                st.session_state['gantt_history'] = gantt_history
                st.session_state['score'] = score
                st.session_state['total_reward'] = total_reward
                st.session_state['show_results'] = True
                
                # 同时保存到持久化变量
                st.session_state['last_stats'] = final_stats
                st.session_state['last_gantt_history'] = gantt_history
                st.session_state['last_score'] = score
                st.session_state['last_total_reward'] = total_reward
                
                save_app_state()  # 💾 保存状态
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
        
        # 甘特图
        with st.expander(get_text("gantt_chart", lang), expanded=True):
            gantt_fig = create_gantt_chart(gantt_history)
            if gantt_fig:
                st.plotly_chart(gantt_fig, use_container_width=True)
                
                # 提供下载选项
                if st.button(get_text("download_gantt", lang)):
                    html_str = gantt_fig.to_html()
                    st.download_button(
                        label=get_text("download_gantt_btn", lang),
                        data=html_str,
                        file_name=f"gantt_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html"
                    )
            else:
                st.warning(get_text("warn_gantt_no_data", lang))
        
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

if __name__ == "__main__":
    setup_page()
    main()
