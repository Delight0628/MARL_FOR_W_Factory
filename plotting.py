import plotly.express as px
import pandas as pd
from typing import List, Dict, Any
import os


_ZH_EN_MAP = {
    "调度甘特图": "Scheduling Gantt Chart",
    "方法": "Method",
    "配置": "Scenario",
    "模拟时间": "Simulated Time",
    "分钟": "min",
    "工作站": "Workstation",
    "产品": "Product",
    "零件ID": "Part ID",
    "订单ID": "Order ID",
    "开始时间": "Start Time",
    "结束时间": "Finish Time",
    "持续时间": "Duration",
    "没有可用于生成甘特图的加工历史数据。": "No processing history data available to generate a Gantt chart.",
    "静态评估": "Static Evaluation",
    "泛化测试1-高压力短交期": "Generalization Test 1 - High Pressure & Short Due Dates",
    "泛化测试2-混合优先级": "Generalization Test 2 - Mixed Priorities",
    "泛化测试3-大批量长周期": "Generalization Test 3 - Large Batch & Long Horizon",
    "黑胡桃木餐桌": "Black Walnut Dining Table",
    "橡木书柜": "Oak Bookcase",
    "松木床架": "Pine Bed Frame",
    "樱桃木椅子": "Cherry Wood Chair",
    "组装台": "Assembly Station",
    "砂光机": "Sanding Machine",
    "带锯机": "Band Saw",
    "包装台": "Packaging Station",
    "五轴加工中心": "5-Axis Machining Center",
}


def _translate_zh_to_en(text: str) -> str:
    if text is None:
        return text
    s = str(text)
    for zh in sorted(_ZH_EN_MAP.keys(), key=len, reverse=True):
        en = _ZH_EN_MAP[zh]
        # 直接替换明文中文
        s = s.replace(zh, en)
        # 同时替换Plotly HTML中常见的unicode转义形式（例如：\u8c03\u5ea6\u7518\u7279\u56fe）
        try:
            zh_escaped = zh.encode("unicode_escape").decode("ascii")
            en_escaped = en.encode("unicode_escape").decode("ascii")
            s = s.replace(zh_escaped, en_escaped)
        except Exception:
            pass
    return s


def translate_plotly_html_inplace(file_path: str) -> bool:
    if not os.path.exists(file_path):
        return False
    if not file_path.lower().endswith(".html"):
        return False
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    new_content = _translate_zh_to_en(content)
    if new_content == content:
        return True
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    return True


def translate_plotly_htmls_under_dir(root_dir: str) -> int:
    if not root_dir or not os.path.isdir(root_dir):
        return 0
    changed = 0
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            if not name.lower().endswith(".html"):
                continue
            fp = os.path.join(dirpath, name)
            ok = translate_plotly_html_inplace(fp)
            if ok:
                changed += 1
    return changed

def generate_gantt_chart(history: List[Dict[str, Any]], method_name: str, config_name: str, output_dir: str = None, run_name: str = None):
    """
    根据加工历史生成交互式甘特图。

    Args:
        history (List[Dict[str, Any]]): 包含零件加工记录的列表。
        method_name (str): 评估的方法名称 (e.g., "MARL (PPO)", "SPT")。
        config_name (str): 测试配置的名称 (e.g., "静态评估")。
        output_dir (str, optional): 保存文件的目录. Defaults to None.
    """
    if not history:
        msg = _translate_zh_to_en("没有可用于生成甘特图的加工历史数据。")
        print(f"[{method_name} - {_translate_zh_to_en(config_name)}] {msg}", flush=True)
        return

    # 将历史数据转换为Pandas DataFrame
    df = pd.DataFrame(history)

    # 将中文资源/产品/配置名翻译为英文，保证图上显示一致
    try:
        if 'Resource' in df.columns:
            df['Resource'] = df['Resource'].map(_translate_zh_to_en)
        if 'Product' in df.columns:
            df['Product'] = df['Product'].map(_translate_zh_to_en)
        if 'Task' in df.columns:
            df['Task'] = df['Task'].map(_translate_zh_to_en)
    except Exception:
        pass
    
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
                         f"Workstation: {row['Resource']}<br>" +
                         f"Product: {row['Product']}<br>" +
                         f"Part ID: {row['Part ID']}<br>" +
                         f"Order ID: {row['Order ID']}<br>" +
                         f"Start Time: {row['Start']:.1f} min<br>" +
                         f"Finish Time: {row['Finish']:.1f} min<br>" +
                         f"Duration: {row['Duration']:.1f} min<extra></extra>",
            name=row['Product'],
            showlegend=row['Product'] not in [trace.name for trace in fig.data]  # 只显示一次图例
        ))
    
    # 更新图表布局
    config_name_en = _translate_zh_to_en(config_name)
    fig.update_layout(
        title=f"Scheduling Gantt Chart - Method: {method_name} | Scenario: {config_name_en}",
        xaxis_title="Simulated Time (min)",
        yaxis_title="Workstation",
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
    safe_method = method_name.replace(' ', '_').replace('(', '').replace(')', '')
    safe_config = config_name.replace(' ', '_').replace('/', '-') # 替换/防止路径问题
    
    if run_name:
        safe_run = run_name.replace(" ", "_").replace("/", "-")
        filename = f"{safe_run}_{safe_method}_{safe_config}.html"
    else:
        # 如果没有提供run_name，则回退到旧的命名方式
        filename = f"gantt_{safe_method}_{safe_config}.html"
    
    # 如果指定了输出目录，则保存到该目录下
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
    else:
        filepath = filename

    fig.write_html(filepath)
    print(f"📊 甘特图已保存至: {filepath}", flush=True)
