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
    calculate_episode_score
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

# 隐藏右上角的Deploy按钮和菜单
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ============================================================================
# 辅助函数
# ============================================================================

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

def run_scheduling(actor_model, orders_config, max_steps=1500):
    """运行调度仿真"""
    config = {
        'custom_orders': orders_config,
        'disable_failures': True,
        'stage_name': '用户自定义调度'
    }
    
    env = WFactoryEnv(config=config)
    obs, info = env.reset(seed=42)
    
    step_count = 0
    total_reward = 0
    
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
        
        if any(terminations.values()) or any(truncations.values()):
            break
    
    final_stats = env.sim.get_final_stats()
    gantt_history = env.sim.gantt_chart_history
    score = calculate_episode_score(final_stats, config)
    
    env.close()
    
    return final_stats, gantt_history, score, total_reward

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
    
    # 步骤2：订单配置
    st.header("📝 订单配置")
    
    # 提供两种配置方式
    config_method = st.radio(
        "选择配置方式",
        ["可视化配置", "JSON配置"],
        horizontal=True
    )
    
    if config_method == "可视化配置":
        st.subheader("添加订单")
        
        # 初始化订单列表
        if 'orders' not in st.session_state:
            st.session_state['orders'] = []
        
        # 添加订单表单
        with st.form("add_order_form"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                product = st.selectbox(
                    "产品类型",
                    options=list(PRODUCT_ROUTES.keys())
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
            
            submitted = st.form_submit_button("➕ 添加订单")
            if submitted:
                order = {
                    "product": product,
                    "quantity": int(quantity),
                    "priority": int(priority),
                    "due_date": float(due_date)
                }
                st.session_state['orders'].append(order)
                st.success(f"已添加订单：{product} x{quantity}")
        
        # 显示当前订单列表
        if st.session_state['orders']:
            st.subheader("当前订单列表")
            
            orders_df = pd.DataFrame(st.session_state['orders'])
            orders_df.index = range(1, len(orders_df) + 1)
            orders_df.columns = ['产品', '数量', '优先级', '交期(分钟)']
            
            st.dataframe(orders_df, use_container_width=True)
            
            # 订单管理按钮
            col1, col2, col3 = st.columns([1, 1, 3])
            with col1:
                if st.button("🗑️ 清空所有订单"):
                    st.session_state['orders'] = []
                    st.rerun()
            
            with col2:
                # 导出订单配置
                if st.button("💾 导出配置"):
                    config_json = json.dumps(st.session_state['orders'], indent=2, ensure_ascii=False)
                    st.download_button(
                        label="下载JSON配置",
                        data=config_json,
                        file_name=f"orders_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            
            # 显示订单统计
            total_parts = sum(order['quantity'] for order in st.session_state['orders'])
            st.info(f"📦 订单总数：{len(st.session_state['orders'])} | 总零件数：{total_parts}")
            
    else:  # JSON配置
        st.subheader("JSON格式配置")
        
        # 提供示例
        example_json = [
            {"product": "黑胡桃木餐桌", "quantity": 6, "priority": 1, "due_date": 300.0},
            {"product": "橡木书柜", "quantity": 6, "priority": 2, "due_date": 400.0},
            {"product": "松木床架", "quantity": 6, "priority": 1, "due_date": 350.0}
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
                    # 验证配置
                    for order in orders:
                        if not all(k in order for k in ['product', 'quantity', 'priority', 'due_date']):
                            st.error("配置格式错误：缺少必要字段")
                            break
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
    
    # 开始调度按钮和结果展示区域
    st.divider()
    
    if 'actor_model' not in st.session_state:
        st.warning("⚠️ 请先在左侧加载模型")
    elif not st.session_state.get('orders', []):
        st.warning("⚠️ 请先配置订单")
    else:
        if st.button("🚀 开始调度仿真", type="primary", use_container_width=True):
            with st.spinner("正在运行调度仿真，请稍候..."):
                try:
                    actor_model = st.session_state['actor_model']
                    orders = st.session_state['orders']
                    
                    final_stats, gantt_history, score, total_reward = run_scheduling(
                        actor_model, orders
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
