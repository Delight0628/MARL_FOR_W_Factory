import plotly.express as px
import pandas as pd
from typing import List, Dict, Any
import os

def generate_gantt_chart(history: List[Dict[str, Any]], method_name: str, config_name: str, output_dir: str = None):
    """
    根据加工历史生成交互式甘特图。

    Args:
        history (List[Dict[str, Any]]): 包含零件加工记录的列表。
        method_name (str): 评估的方法名称 (e.g., "MARL (PPO)", "SPT")。
        config_name (str): 测试配置的名称 (e.g., "静态评估")。
        output_dir (str, optional): 保存文件的目录. Defaults to None.
    """
    if not history:
        print(f"[{method_name} - {config_name}] 没有可用于生成甘特图的加工历史数据。", flush=True)
        return

    # 将历史数据转换为Pandas DataFrame
    df = pd.DataFrame(history)
    
    # 🔧 关键修复：将数值时间转换为字符串，避免被Plotly误认为Unix时间戳
    # 同时创建数值列用于正确排序
    df['Start_Time'] = df['Start'].astype(str) + ' min'
    df['Finish_Time'] = df['Finish'].astype(str) + ' min'
    df['Start_Numeric'] = df['Start']
    df['Finish_Numeric'] = df['Finish']
    
    # 🔧 使用px.bar创建水平条形图来模拟甘特图，避免timeline的时间戳问题
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    # 获取所有唯一的工作站和产品类型
    resources = df['Resource'].unique()
    products = df['Product'].unique()
    
    # 为每种产品类型分配颜色
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    color_map = {product: colors[i % len(colors)] for i, product in enumerate(products)}
    
    # 为每个资源（工作站）添加甘特图条
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
            showlegend=row['Product'] not in [trace.name for trace in fig.data]  # 只显示一次图例
        ))
    
    # 更新图表布局
    fig.update_layout(
        title=f"调度甘特图 - 方法: {method_name} | 配置: {config_name}",
        xaxis_title="模拟时间 (分钟)",
        yaxis_title="工作站",
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="black"
        ),
        hovermode='closest',
        width=1200,
        height=600
    )
    
    # 确保X轴显示为数值而不是时间戳，并按工作站排序
    fig.update_xaxes(type='linear')
    fig.update_yaxes(categoryorder="category ascending")

    # 保存为HTML文件
    filename = f"gantt_{method_name.replace(' ', '_').replace('(', '').replace(')', '')}_{config_name}.html"
    
    # 如果指定了输出目录，则保存到该目录下
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
    else:
        filepath = filename

    fig.write_html(filepath)
    print(f"📊 甘特图已保存至: {filepath}", flush=True)
