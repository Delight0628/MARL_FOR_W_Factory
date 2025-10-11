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

# 禁用GPU，使用CPU运行
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 添加项目路径
app_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(app_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from environments.w_factory_env import WFactoryEnv
from environments.w_factory_config import (
    PRODUCT_ROUTES, WORKSTATIONS, get_total_parts_count,
    calculate_episode_score, generate_random_orders
)

# ============================================================================
# 页面配置
# ============================================================================
st.set_page_config(
    page_title="W工厂智能调度系统",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed"
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
        st.error(f"保存失败：{e}")
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
            'warnings': [f"❌ 以下产品没有定义工艺路线：{', '.join(missing)}"],
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
        difficulty_level = "极高 ⚠️"
        warnings.append(f"⚠️ 理论最短完工时间({theoretical_makespan:.1f}min)超过标准仿真时间({simulation_time}min)，订单可能无法全部完成！")
        warnings.append(f"💡 建议：减少订单数量或延长交期时间")
    elif makespan_ratio > 0.8:
        difficulty_level = "高 🎯"
        warnings.append(f"🎯 高挑战性任务：理论完工时间占仿真时间的{makespan_ratio*100:.1f}%，时间非常紧张")
    elif makespan_ratio > 0.5:
        difficulty_level = "中等 ⚡"
        warnings.append(f"⚡ 中等难度任务：理论完工时间占仿真时间的{makespan_ratio*100:.1f}%，有一定挑战")
    else:
        difficulty_level = "低 ✅"
        warnings.append(f"✅ 任务难度适中：理论完工时间占仿真时间的{makespan_ratio*100:.1f}%")
    
    # 7. 检查交期是否合理
    if min_due_date < theoretical_makespan * 0.5:
        warnings.append(f"⚠️ 部分订单交期过短(最短{min_due_date:.0f}min)，可能导致严重延期")
    
    if theoretical_makespan > avg_due_date:
        warnings.append(f"⚠️ 平均交期({avg_due_date:.0f}min)短于理论完工时间({theoretical_makespan:.1f}min)，大部分订单可能延期")
    
    # 8. 检查瓶颈工作站
    bottleneck_ratio = info['bottleneck_load'] / simulation_time
    if bottleneck_ratio > 0.9:
        warnings.append(f"🔍 瓶颈工作站'{bottleneck_station}'负荷极高({bottleneck_ratio*100:.0f}%)，可能严重影响整体进度")
    elif bottleneck_ratio > 0.7:
        warnings.append(f"🔍 瓶颈工作站'{bottleneck_station}'负荷较高({bottleneck_ratio*100:.0f}%)，需要优化调度策略")
    
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
        if not os.path.exists(model_path):
            return None, f"错误：模型文件不存在 - {model_path}"
        
        actor_model = tf.keras.models.load_model(model_path)
        return actor_model, "模型加载成功！"
    except Exception as e:
        return None, f"加载模型失败：{str(e)}"

def find_available_models(base_dir="mappo/ppo_models"):
    """查找所有可用的训练模型"""
    models = []
    models_path = os.path.join(project_root, base_dir)
    if not os.path.exists(models_path):
        return models
    
    for timestamp_dir in sorted(os.listdir(models_path), reverse=True):
        dir_path = os.path.join(models_path, timestamp_dir)
        if os.path.isdir(dir_path):
            for file in os.listdir(dir_path):
                if file.endswith("_actor.keras"):
                    model_path = os.path.join(dir_path, file)
                    model_name = file.replace("_actor.keras", "")
                    models.append({
                        "name": f"{timestamp_dir}/{model_name}",
                        "path": model_path,
                        "timestamp": timestamp_dir,
                        "type": model_name
                    })
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
            status_text.text("🔄 初始化环境...")
        
        env = WFactoryEnv(config=config)
        obs, info = env.reset(seed=42)
        
        step_count = 0
        total_reward = 0
        
        if status_text:
            status_text.text("🚀 开始调度仿真...")
        
        while step_count < max_steps:
            actions = {}
            for agent in env.agents:
                if agent in obs:
                    state = tf.expand_dims(obs[agent], 0)
                    action_probs = actor_model(state, training=False)
                    action = int(tf.argmax(action_probs[0]))
                    actions[agent] = action
            
            obs, rewards, terminations, truncations, info = env.step(actions)
            total_reward += sum(rewards.values())
            step_count += 1
            
            # 更新进度条
            if progress_bar and step_count % 10 == 0:
                progress = min(step_count / max_steps, 1.0)
                progress_bar.progress(progress)
                if status_text:
                    status_text.text(f"⚙️ 调度中... ({step_count}/{max_steps} 步)")
            
            if any(terminations.values()) or any(truncations.values()):
                break
        
        if status_text:
            status_text.text("📊 生成结果...")
        
        if progress_bar:
            progress_bar.progress(1.0)
        
        final_stats = env.sim.get_final_stats()
        gantt_history = env.sim.gantt_chart_history
        score = calculate_episode_score(final_stats, config)
        
        env.close()
        
        if status_text:
            status_text.text("✅ 调度完成!")
        
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
    
    # 为每个加工任务添加甘特图条
    for _, row in df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['Start'], row['Finish'], row['Finish'], row['Start'], row['Start']],
            y=[row['Resource'], row['Resource'], row['Resource'], row['Resource'], row['Resource']],
            fill='toself',
            fillcolor=color_map[row['Product']],
            line=dict(color=color_map[row['Product']], width=2),
            hovertemplate=f"<b>{row['Task']}</b><br>" +
                         f"工作站: {row['Resource']}<br>" +
                         f"产品: {row['Product']}<br>" +
                         f"零件ID: {row['Part ID']}<br>" +
                         f"订单ID: {row['Order ID']}<br>" +
                         f"开始时间: {row['Start']:.1f}分钟<br>" +
                         f"结束时间: {row['Finish']:.1f}分钟<br>" +
                         f"持续时间: {row['Duration']:.1f}分钟<extra></extra>",
            name=row['Product'],
            showlegend=row['Product'] not in [trace.name for trace in fig.data]
        ))
    
    fig.update_layout(
        title="生产调度甘特图",
        xaxis_title="时间 (分钟)",
        yaxis_title="工作站",
        font=dict(family="Arial, sans-serif", size=12),
        hovermode='closest',
        height=500,
        showlegend=True
    )
    
    fig.update_xaxes(type='linear')
    fig.update_yaxes(categoryorder="category ascending")
    
    return fig

def create_utilization_chart(stats):
    """创建设备利用率柱状图"""
    utilization_data = stats.get('equipment_utilization', {})
    
    if not utilization_data:
        return None
    
    df = pd.DataFrame([
        {"工作站": station, "利用率": util * 100}
        for station, util in utilization_data.items()
    ])
    
    fig = go.Figure(data=[
        go.Bar(
            x=df['工作站'],
            y=df['利用率'],
            text=df['利用率'].apply(lambda x: f"{x:.1f}%"),
            textposition='auto',
            marker_color='steelblue'
        )
    ])
    
    fig.update_layout(
        title="各工作站设备利用率",
        xaxis_title="工作站",
        yaxis_title="利用率 (%)",
        height=400
    )
    
    return fig

# ============================================================================
# 主应用界面
# ============================================================================

def main():
    st.title("🏭 W工厂智能调度系统")
    st.markdown("**基于多智能体强化学习的生产调度优化系统**")
    
    # 步骤1：模型加载
    st.header("⚙️ 系统配置")
    
    # 模型加载方式选择
    model_input_method = st.radio(
        "选择模型加载方式",
        ["从训练历史中选择", "手动输入路径"],
        horizontal=True
    )
    
    actor_model = None
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if model_input_method == "从训练历史中选择":
            available_models = find_available_models()
            
            if not available_models:
                st.warning("未找到已训练的模型，请先训练模型或手动输入路径")
                model_path = None
            else:
                model_options = [m["name"] for m in available_models]
                selected_model = st.selectbox(
                    "选择训练好的模型",
                    options=model_options,
                    help="显示格式：训练时间戳/模型类型"
                )
                
                selected_model_info = next(m for m in available_models if m["name"] == selected_model)
                model_path = selected_model_info["path"]
                
                st.caption(f"📂 模型路径：{model_path}")
        else:
            model_path = st.text_input(
                "模型路径",
                value="mappo/ppo_models/",
                help="输入.keras格式的Actor模型文件完整路径"
            )
    
    with col2:
        st.write("")  # 空行对齐
        st.write("")  # 空行对齐
        # 加载模型按钮
        if st.button("🔄 加载模型", type="primary", use_container_width=True):
            if model_path:
                with st.spinner("正在加载模型..."):
                    actor_model, message = load_model(model_path)
                    if actor_model is not None:
                        st.session_state['actor_model'] = actor_model
                        st.session_state['model_path'] = model_path
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
            else:
                st.error("请先选择或输入模型路径")
    
    # 显示已加载的模型状态
    if 'actor_model' in st.session_state:
        st.success(f"✅ 模型已加载：{st.session_state.get('model_path', '未知')}")
    
    st.divider()
    
    # 自定义产品工艺路线管理（移到系统配置部分）
    with st.expander("🔧 自定义产品工艺路线", expanded=False):
        st.caption("添加新的产品类型并定义其工艺路线（保存后可在订单配置中使用）")
        
        # 初始化自定义产品路线
        if 'custom_products' not in st.session_state:
            st.session_state['custom_products'] = load_custom_products()
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            new_product_name = st.text_input("新产品名称", placeholder="例如：橡木办公桌")
        
        with col2:
            st.write("")  # 空行对齐
        
        st.write("**工艺路线定义**（按加工顺序）")
        
        # 显示可用工作站
        st.caption(f"可用工作站：{', '.join(WORKSTATIONS.keys())}")
        
        # 工艺步骤输入
        num_steps = st.number_input("工序数量", min_value=1, max_value=10, value=3, key="custom_steps")
        
        route_steps = []
        for i in range(num_steps):
            col1, col2 = st.columns([2, 1])
            with col1:
                station = st.selectbox(
                    f"工序 {i+1} - 工作站",
                    options=list(WORKSTATIONS.keys()),
                    key=f"custom_station_{i}"
                )
            with col2:
                time = st.number_input(
                    f"工序 {i+1} - 时间(分钟)",
                    min_value=1,
                    max_value=100,
                    value=10,
                    key=f"custom_time_{i}"
                )
            route_steps.append({"station": station, "time": time})
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("➕ 添加自定义产品"):
                if new_product_name:
                    if new_product_name in PRODUCT_ROUTES:
                        st.error(f"产品 '{new_product_name}' 已存在于系统内置产品中")
                    else:
                        st.session_state['custom_products'][new_product_name] = route_steps
                        save_custom_products(st.session_state['custom_products'])
                        st.success(f"✅ 已添加产品：{new_product_name}")
                        st.rerun()
                else:
                    st.error("请输入产品名称")
        
        # 显示已添加的自定义产品
        if st.session_state['custom_products']:
            st.divider()
            st.write("**已添加的自定义产品：**")
            
            for prod_name, route in st.session_state['custom_products'].items():
                col1, col2 = st.columns([4, 1])
                with col1:
                    route_str = " → ".join([f"{s['station']}({s['time']}min)" for s in route])
                    st.text(f"• {prod_name}: {route_str}")
                with col2:
                    if st.button("🗑️", key=f"del_{prod_name}"):
                        del st.session_state['custom_products'][prod_name]
                        save_custom_products(st.session_state['custom_products'])
                        st.rerun()
    
    st.divider()
    
    # 步骤2：订单配置
    st.header("📝 订单配置")
    
    # 提供三种配置方式
    config_method = st.radio(
        "选择配置方式",
        ["可视化配置", "JSON配置", "随机生成订单"],
        horizontal=True
    )
    
    if config_method == "可视化配置":
        # 初始化订单列表
        if 'orders' not in st.session_state:
            st.session_state['orders'] = []
        
        st.subheader("添加订单")
        
        # 合并系统产品和自定义产品
        custom_products = st.session_state.get('custom_products', {})
        all_products = list(PRODUCT_ROUTES.keys()) + list(custom_products.keys())
        
        # 添加订单表单
        with st.form("add_order_form"):
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                product = st.selectbox(
                    "产品类型",
                    options=all_products
                )
            
            with col2:
                quantity = st.number_input(
                    "数量",
                    min_value=1,
                    max_value=100,
                    value=5
                )
            
            with col3:
                priority = st.number_input(
                    "优先级",
                    min_value=1,
                    max_value=5,
                    value=1,
                    help="1=最高优先级，5=最低优先级"
                )
            
            with col4:
                due_date = st.number_input(
                    "交期(分钟)",
                    min_value=50,
                    max_value=2000,
                    value=300
                )
            
            with col5:
                arrival_time = st.number_input(
                    "到达时间(分钟)",
                    min_value=0,
                    max_value=500,
                    value=0,
                    help="订单到达时间，0表示立即到达"
                )
            
            submitted = st.form_submit_button("➕ 添加订单")
            if submitted:
                order = {
                    "product": product,
                    "quantity": int(quantity),
                    "priority": int(priority),
                    "due_date": int(due_date),
                    "arrival_time": int(arrival_time)
                }
                st.session_state['orders'].append(order)
                st.success(f"已添加订单：{product} x{quantity} (到达时间:{arrival_time}min)")
                st.rerun()
    
    elif config_method == "JSON配置":
        st.subheader("JSON格式配置")
        
        # 提供示例
        example_json = [
            {"product": "黑胡桃木餐桌", "quantity": 6, "priority": 1, "due_date": 300, "arrival_time": 0},
            {"product": "橡木书柜", "quantity": 6, "priority": 2, "due_date": 400, "arrival_time": 0},
            {"product": "松木床架", "quantity": 6, "priority": 1, "due_date": 350, "arrival_time": 20}
        ]
        
        st.caption("示例格式：")
        st.code(json.dumps(example_json, indent=2, ensure_ascii=False), language="json")
        
        json_input = st.text_area(
            "输入订单配置（JSON格式）",
            height=300,
            help="请输入符合格式的JSON配置"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ 加载JSON配置"):
                try:
                    orders = json.loads(json_input)
                    # 验证配置并添加默认值
                    for order in orders:
                        if not all(k in order for k in ['product', 'quantity', 'priority', 'due_date']):
                            st.error("配置格式错误：缺少必要字段(product, quantity, priority, due_date)")
                            break
                        # 添加默认arrival_time
                        if 'arrival_time' not in order:
                            order['arrival_time'] = 0
                        # 确保交期和到达时间是整数
                        order['due_date'] = int(order['due_date'])
                        order['arrival_time'] = int(order['arrival_time'])
                    else:
                        st.session_state['orders'] = orders
                        st.success(f"成功加载 {len(orders)} 个订单")
                        st.rerun()
                except json.JSONDecodeError as e:
                    st.error(f"JSON格式错误：{str(e)}")
        
        with col2:
            if st.button("📋 使用示例配置"):
                st.session_state['orders'] = example_json
                st.success("已加载示例配置")
                st.rerun()
    
    else:  # 随机生成订单
        st.subheader("随机订单生成")
        
        col1, col2 = st.columns(2)
        
        with col1:
            num_orders = st.slider("订单数量", min_value=3, max_value=10, value=5)
            min_quantity = st.number_input("每个订单最小零件数", min_value=1, max_value=20, value=3)
            max_quantity = st.number_input("每个订单最大零件数", min_value=1, max_value=50, value=10)
        
        with col2:
            min_due = st.number_input("最短交期(分钟)", min_value=100, max_value=1000, value=200)
            max_due = st.number_input("最长交期(分钟)", min_value=200, max_value=2000, value=700)
        
        if st.button("🎲 生成随机订单", type="primary"):
            # 自定义配置
            config = {
                "min_orders": num_orders,
                "max_orders": num_orders,
                "min_quantity_per_order": min_quantity,
                "max_quantity_per_order": max_quantity,
                "due_date_range": (min_due, max_due),
                "priority_weights": [0.3, 0.5, 0.2]
            }
            
            # 临时修改全局配置
            from environments import w_factory_config
            import random
            original_config = w_factory_config.TRAINING_FLOW_CONFIG["generalization_phase"]["random_orders_config"]
            w_factory_config.TRAINING_FLOW_CONFIG["generalization_phase"]["random_orders_config"] = config
            
            try:
                random_orders = generate_random_orders()
                # 修正：确保交期是整数，并添加随机到达时间
                for order in random_orders:
                    order['due_date'] = int(order['due_date'])
                    order['arrival_time'] = int(random.uniform(0, 50))  # 0-50分钟的随机到达时间
                st.session_state['orders'] = random_orders
                st.success(f"✅ 已生成 {len(random_orders)} 个随机订单")
                st.rerun()
            finally:
                # 恢复原配置
                w_factory_config.TRAINING_FLOW_CONFIG["generalization_phase"]["random_orders_config"] = original_config
    
    # 显示当前订单列表（所有模式通用）
    if st.session_state.get('orders'):
        st.divider()
        st.subheader("📋 当前订单列表")
        
        orders_df = pd.DataFrame(st.session_state['orders'])
        orders_df.index = range(1, len(orders_df) + 1)
        
        # 根据列数设置列名
        if len(orders_df.columns) == 5:
            orders_df.columns = ['产品', '数量', '优先级', '交期(分钟)', '到达时间(分钟)']
        else:
            orders_df.columns = ['产品', '数量', '优先级', '交期(分钟)']
        
        st.dataframe(orders_df, use_container_width=True)
        
        # 订单管理按钮
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            if st.button("🗑️ 清空订单"):
                st.session_state['orders'] = []
                st.rerun()
        
        with col2:
            config_json = json.dumps(st.session_state['orders'], indent=2, ensure_ascii=False)
            st.download_button(
                label="💾 导出配置",
                data=config_json,
                file_name=f"orders_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        # 显示订单统计
        total_parts = sum(order['quantity'] for order in st.session_state['orders'])
        st.caption(f"📦 订单总数：{len(st.session_state['orders'])} | 总零件数：{total_parts}")
        
        # 🔧 新增：订单配置合理性检测
        st.divider()
        st.subheader("🔍 订单配置分析")
        
        custom_products = st.session_state.get('custom_products', {})
        validation_result = validate_order_config(st.session_state['orders'], custom_products)
        
        if not validation_result['valid']:
            st.error("❌ 订单配置无效")
            for warning in validation_result['warnings']:
                st.warning(warning)
        else:
            info = validation_result['info']
            
            # 显示难度评估
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("任务难度", validation_result['difficulty_level'])
            with col2:
                st.metric("总零件数", f"{info['total_parts']}")
            with col3:
                st.metric("理论完工时间", f"{info['theoretical_makespan']:.0f}min")
            with col4:
                st.metric("瓶颈工作站", info['bottleneck_station'])
            
            # 显示详细信息和警告
            with st.expander("📊 查看详细分析", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**基础统计**")
                    st.write(f"- 总加工时间：{info['total_processing_time']:.1f} 分钟")
                    st.write(f"- 平均交期：{info['avg_due_date']:.0f} 分钟")
                    st.write(f"- 最短交期：{info['min_due_date']:.0f} 分钟")
                    st.write(f"- 最长交期：{info['max_due_date']:.0f} 分钟")
                    if info.get('has_arrival_time'):
                        st.write(f"- 最晚到达：{info['max_arrival_time']:.0f} 分钟")
                
                with col2:
                    st.write("**瓶颈分析**")
                    st.write(f"- 瓶颈工作站：{info['bottleneck_station']}")
                    st.write(f"- 瓶颈负荷：{info['bottleneck_load']:.1f} 分钟")
                    st.write(f"- 负荷率：{info['bottleneck_load']/SIMULATION_TIME*100:.1f}%")
                    st.write(f"- 标准仿真时间：{SIMULATION_TIME} 分钟")
                
                # 显示警告和建议
                if validation_result['warnings']:
                    st.write("**⚠️ 提示与建议**")
                    for warning in validation_result['warnings']:
                        st.write(f"- {warning}")
    
    # 开始调度按钮和结果展示区域
    st.divider()
    
    if 'actor_model' not in st.session_state:
        st.warning("⚠️ 请先在上方加载模型")
    elif not st.session_state.get('orders', []):
        st.warning("⚠️ 请先配置订单")
    else:
        if st.button("🚀 开始调度仿真", type="primary", use_container_width=True):
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
                
                st.success("✅ 调度仿真完成！")
                st.rerun()
                
            except Exception as e:
                st.error(f"调度仿真失败：{str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # 显示调度结果（在按钮下方）
    if st.session_state.get('show_results', False) and 'final_stats' in st.session_state:
        st.divider()
        st.header("📊 调度结果")
        
        stats = st.session_state['final_stats']
        gantt_history = st.session_state['gantt_history']
        score = st.session_state['score']
        total_reward = st.session_state['total_reward']
        orders = st.session_state['orders']
        
        # KPI指标展示
        st.subheader("📈 关键绩效指标（KPI）")
        
        total_parts_target = sum(order["quantity"] for order in orders)
        completion_rate = (stats['total_parts'] / total_parts_target) * 100 if total_parts_target > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📦 完成零件数",
                value=f"{stats['total_parts']}/{total_parts_target}",
                delta=f"{completion_rate:.1f}%"
            )
        
        with col2:
            st.metric(
                label="⏱️ 总完工时间",
                value=f"{stats['makespan']:.1f} 分钟"
            )
        
        with col3:
            st.metric(
                label="📊 设备利用率",
                value=f"{stats['mean_utilization']*100:.1f}%"
            )
        
        with col4:
            st.metric(
                label="⚠️ 订单延期",
                value=f"{stats['total_tardiness']:.1f} 分钟"
            )
        
        # 综合评分
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="🎯 综合评分",
                value=f"{score:.3f}",
                help="基于完成率、延期、完工时间和利用率的综合评分"
            )
        with col2:
            st.metric(
                label="💰 累计奖励",
                value=f"{total_reward:.1f}"
            )
        
        # 设备利用率图表
        with st.expander("🔧 设备利用率分析", expanded=True):
            util_chart = create_utilization_chart(stats)
            if util_chart:
                st.plotly_chart(util_chart, use_container_width=True)
        
        # 甘特图
        with st.expander("📊 调度甘特图", expanded=True):
            gantt_fig = create_gantt_chart(gantt_history)
            if gantt_fig:
                st.plotly_chart(gantt_fig, use_container_width=True)
                
                # 提供下载选项
                if st.button("💾 下载甘特图HTML"):
                    html_str = gantt_fig.to_html()
                    st.download_button(
                        label="下载",
                        data=html_str,
                        file_name=f"gantt_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html"
                    )
            else:
                st.warning("无法生成甘特图：没有加工历史数据")
        
        # 详细统计信息
        with st.expander("📋 详细统计信息"):
            st.json({
                "完成零件数": stats['total_parts'],
                "总完工时间(分钟)": stats['makespan'],
                "设备平均利用率": f"{stats['mean_utilization']*100:.2f}%",
                "总延期时间(分钟)": stats['total_tardiness'],
                "最大延期时间(分钟)": stats.get('max_tardiness', 0),
                "设备利用率明细": {k: f"{v*100:.2f}%" for k, v in stats.get('equipment_utilization', {}).items()}
            })

if __name__ == "__main__":
    main()
