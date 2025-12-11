"""
多语言支持配置文件
支持：简体中文、繁体中文、英文
"""

LANGUAGES = {
    "zh-CN": "简体中文",
    "zh-TW": "繁體中文",
    "en": "English"
}

# 所有界面文本的多语言版本
TEXTS = {
    # 页面标题和基本信息
    "app_title": {
        "zh-CN": "🏭 W工厂智能调度系统",
        "zh-TW": "🏭 W工廠智能調度系統",
        "en": "🏭 W-Factory Intelligent Scheduling System"
    },
    "page_title": {
        "zh-CN": "W工厂智能调度系统",
        "zh-TW": "W工廠智能調度系統",
        "en": "W-Factory Intelligent Scheduling System"
    },
    "app_subtitle": {
        "zh-CN": "基于多智能体强化学习的生产调度优化系统",
        "zh-TW": "基於多智能體強化學習的生產調度優化系統",
        "en": "Multi-Agent Reinforcement Learning based Production Scheduling Optimization System"
    },
    
    # 系统配置
    "system_config": {
        "zh-CN": "⚙️ 系统配置",
        "zh-TW": "⚙️ 系統配置",
        "en": "⚙️ System Configuration"
    },
    "model_loading_method": {
        "zh-CN": "选择模型加载方式",
        "zh-TW": "選擇模型加載方式",
        "en": "Select Model Loading Method"
    },
    "from_history": {
        "zh-CN": "从训练历史中选择",
        "zh-TW": "從訓練歷史中選擇",
        "en": "Select from Training History"
    },
    "manual_input": {
        "zh-CN": "手动输入路径",
        "zh-TW": "手動輸入路徑",
        "en": "Manual Input Path"
    },
    "select_model": {
        "zh-CN": "选择训练好的模型",
        "zh-TW": "選擇訓練好的模型",
        "en": "Select Trained Model"
    },
    "model_help": {
        "zh-CN": "显示格式：训练时间戳/模型类型",
        "zh-TW": "顯示格式：訓練時間戳/模型類型",
        "en": "Format: Training Timestamp/Model Type"
    },
    "model_path": {
        "zh-CN": "📂 模型路径：",
        "zh-TW": "📂 模型路徑：",
        "en": "📂 Model Path: "
    },
    "model_path_input": {
        "zh-CN": "模型路径",
        "zh-TW": "模型路徑",
        "en": "Model Path"
    },
    "model_path_help": {
        "zh-CN": "输入.keras格式的Actor模型文件完整路径",
        "zh-TW": "輸入.keras格式的Actor模型文件完整路徑",
        "en": "Enter the full path of Actor model file (.keras format)"
    },
    "load_model": {
        "zh-CN": "🔄 加载模型",
        "zh-TW": "🔄 加載模型",
        "en": "🔄 Load Model"
    },
    "model_loaded": {
        "zh-CN": "✅ 模型已加载：",
        "zh-TW": "✅ 模型已加載：",
        "en": "✅ Model Loaded: "
    },
    "no_model_found": {
        "zh-CN": "未找到已训练的模型，请先训练模型或手动输入路径",
        "zh-TW": "未找到已訓練的模型，請先訓練模型或手動輸入路徑",
        "en": "No trained models found. Please train a model first or manually input the path."
    },
    "select_model_first": {
        "zh-CN": "请先选择或输入模型路径",
        "zh-TW": "請先選擇或輸入模型路徑",
        "en": "Please select or input model path first"
    },
    "error_missing_route": {
        "zh-CN": "❌ 以下产品没有定义工艺路线：{}",
        "zh-TW": "❌ 以下產品沒有定義工藝路線：{}",
        "en": "❌ The following products do not have a defined process route: {}"
    },
    "model_loaded_successfully": {
        "zh-CN": "✅ 模型加载成功！",
        "zh-TW": "✅ 模型加載成功！",
        "en": "✅ Model loaded successfully!"
    },
    "error_model_not_found": {
        "zh-CN": "错误：模型文件不存在 - {}",
        "zh-TW": "錯誤：模型文件不存在 - {}",
        "en": "Error: Model file not found - {}"
    },
    "error_load_model_failed": {
        "zh-CN": "加载模型失败：{}",
        "zh-TW": "加載模型失敗：{}",
        "en": "Failed to load model: {}"
    },
    "save_failed": {
        "zh-CN": "保存失败：{}",
        "zh-TW": "保存失敗：{}",
        "en": "Save failed: {}"
    },
    
    # 自定义产品
    "custom_products": {
        "zh-CN": "🔧 自定义产品工艺路线",
        "zh-TW": "🔧 自定義產品工藝路線",
        "en": "🔧 Custom Product Process Routes"
    },
    "custom_products_caption": {
        "zh-CN": "添加新的产品类型并定义其工艺路线（保存后可在订单配置中使用）",
        "zh-TW": "添加新的產品類型並定義其工藝路線（保存後可在訂單配置中使用）",
        "en": "Add new product types and define their process routes (available in order configuration after saving)"
    },
    "new_product_name": {
        "zh-CN": "新产品名称",
        "zh-TW": "新產品名稱",
        "en": "New Product Name"
    },
    "new_product_placeholder": {
        "zh-CN": "例如：橡木办公桌",
        "zh-TW": "例如：橡木辦公桌",
        "en": "e.g., Oak Office Desk"
    },
    "step_label": {
        "zh-CN": "工序",
        "zh-TW": "工序",
        "en": "Step"
    },
    "workstation_label": {
        "zh-CN": "工作站",
        "zh-TW": "工作站",
        "en": "Workstation"
    },
    "time_label": {
        "zh-CN": "时间(分钟)",
        "zh-TW": "時間(分鐘)",
        "en": "Time (minutes)"
    },
    "process_route": {
        "zh-CN": "工艺路线定义",
        "zh-TW": "工藝路線定義",
        "en": "Process Route Definition"
    },
    "process_order": {
        "zh-CN": "按加工顺序",
        "zh-TW": "按加工順序",
        "en": "In Processing Order"
    },
    "num_steps": {
        "zh-CN": "工序数量",
        "zh-TW": "工序數量",
        "en": "Number of Workstation"
    },
    "step_workstation": {
        "zh-CN": "工序 {} - 工作站",
        "zh-TW": "工序 {} - 工作站",
        "en": "Step {} - Workstation"
    },
    "step_time": {
        "zh-CN": "工序 {} - 时间(分钟)",
        "zh-TW": "工序 {} - 時間(分鐘)",
        "en": "Step {} - Time (minutes)"
    },
    "add_product": {
        "zh-CN": "➕ 添加自定义产品",
        "zh-TW": "➕ 添加自定義產品",
        "en": "➕ Add Custom Product"
    },
    "product_exists": {
        "zh-CN": "产品已存在，请使用不同的名称",
        "zh-TW": "產品已存在，請使用不同的名稱",
        "en": "Product already exists. Please use a different name."
    },
    "product_added": {
        "zh-CN": "✅ 已添加自定义产品：",
        "zh-TW": "✅ 已添加自定義產品：",
        "en": "✅ Custom product added: "
    },
    "added_custom_products": {
        "zh-CN": "**已添加的自定义产品：**",
        "zh-TW": "**已添加的自定義產品：**",
        "en": "**Added Custom Products:**"
    },
    "process_route_definition": {
        "zh-CN": "**工艺路线定义**（按加工顺序）",
        "zh-TW": "**工藝路線定義**（按加工順序）",
        "en": "**Process Route Definition** (in order)"
    },
    "available_workstations": {
        "zh-CN": "可用工作站：{}",
        "zh-TW": "可用工作站：{}",
        "en": "Available Workstations: {}"
    },
    "error_product_already_exists_system": {
        "zh-CN": "产品 '{}' 已存在于系统内置产品中",
        "zh-TW": "產品 '{}' 已存在於系統內置產品中",
        "en": "Product '{}' already exists in system products"
    },
    "success_product_added": {
        "zh-CN": "✅ 已添加产品：{}",
        "zh-TW": "✅ 已添加產品：{}",
        "en": "✅ Product added: {}"
    },
    "error_enter_product_name": {
        "zh-CN": "请输入产品名称",
        "zh-TW": "請輸入產品名稱",
        "en": "Please enter a product name"
    },
    
    # 工作站名称
    "带锯机": {
        "zh-CN": "带锯机",
        "zh-TW": "帶鋸機",
        "en": "Sawing Machine"
    },
    "五轴加工中心": {
        "zh-CN": "五轴加工中心",
        "zh-TW": "五軸加工中心",
        "en": "5-Axis Machining Center"
    },
    "砂光机": {
        "zh-CN": "砂光机",
        "zh-TW": "砂光機",
        "en": "Sanding Machine"
    },
    "组装台": {
        "zh-CN": "组装台",
        "zh-TW": "組裝台",
        "en": "Assembly Station"
    },
    "包装台": {
        "zh-CN": "包装台",
        "zh-TW": "包裝台",
        "en": "Packaging Station"
    },

    # 清除配置
    "clear_config": {
        "zh-CN": "🗑️ 清除保存的配置",
        "zh-TW": "🗑️ 清除保存的配置",
        "en": "🗑️ Clear Saved Configuration"
    },
    "clear_config_caption": {
        "zh-CN": "清除所有保存的配置和仿真结果",
        "zh-TW": "清除所有保存的配置和仿真結果",
        "en": "Clear all saved configurations and simulation results"
    },
    "clear_config_help": {
        "zh-CN": "清空所有配置\nClear all saved configurations",
        "zh-TW": "清空所有配置\nClear all saved configurations",
        "en": "Clear all saved configurations"
    },
    "clear_config_error": {
        "zh-CN": "❌ 清除配置失败：{}",
        "zh-TW": "❌ 清除配置失敗：{}",
        "en": "❌ Failed to clear configuration: {}"
    },
    "clear_all": {
        "zh-CN": "🗑️ 清除所有保存",
        "zh-TW": "🗑️ 清除所有保存",
        "en": "🗑️ Clear All Saved"
    },
    "clear_success": {
        "zh-CN": "✅ 已清除所有保存的配置和结果",
        "zh-TW": "✅ 已清除所有保存的配置和結果",
        "en": "✅ All saved configurations and results have been cleared"
    },
    "clear_warning": {
        "zh-CN": "此操作不可逆！",
        "zh-TW": "此操作不可逆！",
        "en": "This action is irreversible!"
    },
    
    # 订单配置
    "order_config": {
        "zh-CN": "📝 订单配置",
        "zh-TW": "📝 訂單配置",
        "en": "📝 Order Configuration"
    },
    "choose_config_method": {
        "zh-CN": "选择配置方式",
        "zh-TW": "選擇配置方式",
        "en": "Choose Configuration Method"
    },
    "product_type": {
        "zh-CN": "产品类型",
        "zh-TW": "產品類型",
        "en": "Product Type"
    },
    "priority_help": {
        "zh-CN": "1=最高优先级，5=最低优先级",
        "zh-TW": "1=最高優先級，5=最低優先級",
        "en": "1=Highest priority, 5=Lowest priority"
    },
    "random_orders": {
        "zh-CN": "随机生成订单",
        "zh-TW": "隨機生成訂單",
        "en": "Random Orders"
    },
    "custom_orders": {
        "zh-CN": "自定义订单",
        "zh-TW": "自定義訂單",
        "en": "Custom Orders"
    },
    "add_order": {
        "zh-CN": "添加订单",
        "zh-TW": "添加訂單",
        "en": "Add Order"
    },
    "product": {
        "zh-CN": "产品",
        "zh-TW": "產品",
        "en": "Product"
    },
    "quantity": {
        "zh-CN": "数量",
        "zh-TW": "數量",
        "en": "Quantity"
    },
    "priority": {
        "zh-CN": "优先级",
        "zh-TW": "優先級",
        "en": "Priority"
    },
    "arrival_time": {
        "zh-CN": "到达时间(分钟)",
        "zh-TW": "到達時間(分鐘)",
        "en": "Arrival Time (min)"
    },
    "arrival_time_help": {
        "zh-CN": "订单到达时间，0表示生产前到达",
        "zh-TW": "訂單到達時間，0表示生產前到達",
        "en": "Order arrival time, 0 means arrival before production"
    },
    "due_date": {
        "zh-CN": "交期(分钟)",
        "zh-TW": "交期(分鐘)",
        "en": "Due Date (min)"
    },
    "add_order_button": {
        "zh-CN": "➕ 添加订单",
        "zh-TW": "➕ 添加訂單",
        "en": "➕ Add Order"
    },
    "order_added": {
        "zh-CN": "已添加订单：{} x{} (到达时间:{}min, 交期:{}min)",
        "zh-TW": "已添加訂單：{} x{} (到達時間:{}min, 交期:{}min)",
        "en": "Order added: {} x{} (Arrival: {}min, Due: {}min)"
    },
    "order_added_full": {
        "zh-CN": "已添加订单：{} x{} (到达时间:{}min, 交期:{}min)",
        "zh-TW": "已添加訂單：{} x{} (到達時間:{}min, 交期:{}min)",
        "en": "Order added: {} x{} (Arrival:{}min, Due:{}min)"
    },
    
    # 随机订单生成
    "random_order_gen": {
        "zh-CN": "随机订单生成",
        "zh-TW": "隨機訂單生成",
        "en": "Random Order Generation"
    },
    "order_count": {
        "zh-CN": "订单数量",
        "zh-TW": "訂單數量",
        "en": "Order Count"
    },
    "product_quantity_range": {
        "zh-CN": "**每个订单的产品数量**",
        "zh-TW": "**每個訂單的產品數量**",
        "en": "**Product Quantity per Order**"
    },
    "due_date_range": {
        "zh-CN": "**交期时间(分钟)**",
        "zh-TW": "**交期時間(分鐘)**",
        "en": "**Due Date (minutes)**"
    },
    "arrival_time_range": {
        "zh-CN": "**到达时间(分钟)**",
        "zh-TW": "**到達時間(分鐘)**",
        "en": "**Arrival Time (minutes)**"
    },
    "from": {
        "zh-CN": "从",
        "zh-TW": "從",
        "en": "From"
    },
    "to": {
        "zh-CN": "到",
        "zh-TW": "到",
        "en": "To"
    },
    "generate_random": {
        "zh-CN": "🎲 生成随机订单",
        "zh-TW": "🎲 生成隨機訂單",
        "en": "🎲 Generate Random Orders"
    },
    "random_generated": {
        "zh-CN": "✅ 已生成 {} 个随机订单",
        "zh-TW": "✅ 已生成 {} 個隨機訂單",
        "en": "✅ Generated {} random orders"
    },
    
    # 订单列表
    "current_orders": {
        "zh-CN": "📋 当前订单列表",
        "zh-TW": "📋 當前訂單列表",
        "en": "📋 Current Order List"
    },
    "order_list_columns_5": {
        "zh-CN": "['产品', '数量', '优先级', '到达时间(分钟)', '交期(分钟)']",
        "zh-TW": "['產品', '數量', '優先級', '到達時間(分鐘)', '交期(分鐘)']",
        "en": "['Product', 'Quantity', 'Priority', 'Arrival Time(min)', 'Due Date(min)']"
    },
    "order_list_columns_4": {
        "zh-CN": "['产品', '数量', '优先级', '交期(分钟)']",
        "zh-TW": "['產品', '數量', '優先級', '交期(分鐘)']",
        "en": "['Product', 'Quantity', 'Priority', 'Due Date(min)']"
    },
    "clear_orders": {
        "zh-CN": "🗑️ 清空订单",
        "zh-TW": "🗑️ 清空訂單",
        "en": "🗑️ Clear Orders"
    },
    "export_config": {
        "zh-CN": "💾 导出配置",
        "zh-TW": "💾 導出配置",
        "en": "💾 Export Config"
    },
    "order_summary": {
        "zh-CN": "📦 订单总数：{} | 产品总数：{}",
        "zh-TW": "📦 訂單總數：{} | 產品總數：{}",
        "en": "📦 Total Orders: {} | Total Products: {}"
    },
    
    # 订单分析
    "order_analysis": {
        "zh-CN": "🔍 订单配置分析",
        "zh-TW": "🔍 訂單配置分析",
        "en": "🔍 Order Configuration Analysis"
    },
    "config_invalid": {
        "zh-CN": "❌ 订单配置无效",
        "zh-TW": "❌ 訂單配置無效",
        "en": "❌ Invalid Order Configuration"
    },
    "config_valid": {
        "zh-CN": "✅ 订单配置有效",
        "zh-TW": "✅ 訂單配置有效",
        "en": "✅ Valid Order Configuration"
    },
    "difficulty": {
        "zh-CN": "任务难度",
        "zh-TW": "任務難度",
        "en": "Task Difficulty"
    },
    "total_parts": {
        "zh-CN": "总零件数",
        "zh-TW": "總零件數",
        "en": "Total Parts"
    },
    "theory_time": {
        "zh-CN": "理论完工时间",
        "zh-TW": "理論完工時間",
        "en": "Theoretical Completion Time"
    },
    "bottleneck": {
        "zh-CN": "瓶颈工作站",
        "zh-TW": "瓶頸工作站",
        "en": "Bottleneck Workstation"
    },
    "view_details": {
        "zh-CN": "📊 查看详细分析",
        "zh-TW": "📊 查看詳細分析",
        "en": "📊 View Detailed Analysis"
    },
    "total_products": {
        "zh-CN": "产品总数",
        "zh-TW": "產品總數",
        "en": "Total Products"
    },
    "basic_stats": {
        "zh-CN": "**基础统计**",
        "zh-TW": "**基礎統計**",
        "en": "**Basic Statistics**"
    },
    "total_processing_time_label": {
        "zh-CN": "- 总加工时间：{:.1f} 分钟",
        "zh-TW": "- 總加工時間：{:.1f} 分鐘",
        "en": "- Total Processing Time: {:.1f} minutes"
    },
    "avg_due_date_label": {
        "zh-CN": "- 平均交期：{:.0f} 分钟",
        "zh-TW": "- 平均交期：{:.0f} 分鐘",
        "en": "- Average Due Date: {:.0f} minutes"
    },
    "min_due_date_label": {
        "zh-CN": "- 最短交期：{:.0f} 分钟",
        "zh-TW": "- 最短交期：{:.0f} 分鐘",
        "en": "- Minimum Due Date: {:.0f} minutes"
    },
    "max_due_date_label": {
        "zh-CN": "- 最长交期：{:.0f} 分钟",
        "zh-TW": "- 最長交期：{:.0f} 分鐘",
        "en": "- Maximum Due Date: {:.0f} minutes"
    },
    "max_arrival_time_label": {
        "zh-CN": "- 最晚到达：{:.0f} 分钟",
        "zh-TW": "- 最晚到達：{:.0f} 分鐘",
        "en": "- Latest Arrival: {:.0f} minutes"
    },
    "bottleneck_analysis": {
        "zh-CN": "**瓶颈分析**",
        "zh-TW": "**瓶頸分析**",
        "en": "**Bottleneck Analysis**"
    },
    "bottleneck_station_label": {
        "zh-CN": "- 瓶颈工作站：{}",
        "zh-TW": "- 瓶頸工作站：{}",
        "en": "- Bottleneck Workstation: {}"
    },
    "bottleneck_load_label": {
        "zh-CN": "- 瓶颈负荷：{:.1f} 分钟",
        "zh-TW": "- 瓶頸負荷：{:.1f} 分鐘",
        "en": "- Bottleneck Load: {:.1f} minutes"
    },
    "load_ratio_label": {
        "zh-CN": "- 负荷率：{:.1f}%",
        "zh-TW": "- 負荷率：{:.1f}%",
        "en": "- Load Ratio: {:.1f}%"
    },
    "standard_simulation_time_label": {
        "zh-CN": "- 标准仿真时间：{} 分钟",
        "zh-TW": "- 標準仿真時間：{} 分鐘",
        "en": "- Standard Simulation Time: {} minutes"
    },
    "tips_and_suggestions": {
        "zh-CN": "**⚠️ 提示与建议**",
        "zh-TW": "**⚠️ 提示與建議**",
        "en": "**⚠️ Tips and Suggestions**"
    },
    
    # 仿真按钮
    "start_simulation": {
        "zh-CN": "🚀 开始调度仿真",
        "zh-TW": "🚀 開始調度仿真",
        "en": "🚀 Start Scheduling Simulation"
    },
    "load_model_first": {
        "zh-CN": "⚠️ 请先在上方加载模型",
        "zh-TW": "⚠️ 請先在上方加載模型",
        "en": "⚠️ Please load model first"
    },
    "config_orders_first": {
        "zh-CN": "⚠️ 请先配置生产订单",
        "zh-TW": "⚠️ 請先配置生產訂單",
        "en": "⚠️ Please configure orders of production first"
    },
    "simulation_complete": {
        "zh-CN": "✅ 调度仿真完成！",
        "zh-TW": "✅ 調度仿真完成！",
        "en": "✅ Scheduling simulation completed!"
    },
    "simulation_failed": {
        "zh-CN": "调度仿真失败：",
        "zh-TW": "調度仿真失敗：",
        "en": "Simulation failed: "
    },
    "generating_results": {
        "zh-CN": "📊 生成结果...",
        "zh-TW": "📊 生成結果...",
        "en": "📊 Generating results..."
    },
    "scheduling_complete": {
        "zh-CN": "✅ 调度完成!",
        "zh-TW": "✅ 調度完成!",
        "en": "✅ Scheduling complete!"
    },
    
    # 结果显示
    "results": {
        "zh-CN": "📊 调度结果",
        "zh-TW": "📊 調度結果",
        "en": "📊 Scheduling Results"
    },
    "kpi": {
        "zh-CN": "📈 关键绩效指标（KPI）",
        "zh-TW": "📈 關鍵績效指標（KPI）",
        "en": "📈 Key Performance Indicators (KPI)"
    },
    "completed_products": {
        "zh-CN": "📦 完成产品数",
        "zh-TW": "📦 完成產品數",
        "en": "📦 Completed Products"
    },
    "makespan": {
        "zh-CN": "⏱️ 总完工时间",
        "zh-TW": "⏱️ 總完工時間",
        "en": "⏱️ Total Makespan"
    },
    "utilization": {
        "zh-CN": "📊 设备利用率",
        "zh-TW": "📊 設備利用率",
        "en": "📊 Equipment Utilization"
    },
    "tardiness": {
        "zh-CN": "⚠️ 订单延期",
        "zh-TW": "⚠️ 訂單延期",
        "en": "⚠️ Order Tardiness"
    },
    "score": {
        "zh-CN": "🎯 综合评分",
        "zh-TW": "🎯 綜合評分",
        "en": "🎯 Comprehensive Score"
    },
    "reward": {
        "zh-CN": "💰 累计奖励",
        "zh-TW": "💰 累計獎勵",
        "en": "💰 Cumulative Reward"
    },
    "score_help": {
        "zh-CN": "基于完成率、延期、完工时间和利用率的综合评分",
        "zh-TW": "基於完成率、延期、完工時間和利用率的綜合評分",
        "en": "Comprehensive score based on completion rate, tardiness, makespan, and utilization"
    },
    "util_analysis": {
        "zh-CN": "🔧 设备利用率分析",
        "zh-TW": "🔧 設備利用率分析",
        "en": "🔧 Equipment Utilization Analysis"
    },
    "gantt_chart": {
        "zh-CN": "📊 调度甘特图",
        "zh-TW": "📊 調度甘特圖",
        "en": "📊 Scheduling Gantt Chart"
    },
    "download_gantt": {
        "zh-CN": "💾 下载甘特图HTML",
        "zh-TW": "💾 下載甘特圖HTML",
        "en": "💾 Download Gantt HTML"
    },
    "download_gantt_btn": {
        "zh-CN": "📥 下载",
        "zh-TW": "📥 下載",
        "en": "📥 Download"
    },
    "warn_gantt_no_data": {
        "zh-CN": "无法生成甘特图：没有加工历史数据",
        "zh-TW": "無法生成甘特圖：沒有加工歷史數據",
        "en": "Cannot generate Gantt chart: no processing history data"
    },
    "detailed_stats": {
        "zh-CN": "📋 详细统计信息",
        "zh-TW": "📋 詳細統計信息",
        "en": "📋 Detailed Statistics"
    },
    "completed_parts_json": {
        "zh-CN": "完成产品数",
        "zh-TW": "完成產品數",
        "en": "Completed Products"
    },
    "makespan_json": {
        "zh-CN": "总完工时间(分钟)",
        "zh-TW": "總完工時間(分鐘)",
        "en": "Total Makespan (minutes)"
    },
    "mean_util_json": {
        "zh-CN": "设备平均利用率",
        "zh-TW": "設備平均利用率",
        "en": "Mean Equipment Utilization"
    },
    "total_tardiness_json": {
        "zh-CN": "总延期时间(分钟)",
        "zh-TW": "總延期時間(分鐘)",
        "en": "Total Tardiness (minutes)"
    },
    "max_tardiness_json": {
        "zh-CN": "最大延期时间(分钟)",
        "zh-TW": "最大延期時間(分鐘)",
        "en": "Max Tardiness (minutes)"
    },
    "util_details_json": {
        "zh-CN": "设备利用率明细",
        "zh-TW": "設備利用率明細",
        "en": "Equipment Utilization Details"
    },
    "gantt_chart_title": {
        "zh-CN": "生产调度甘特图",
        "zh-TW": "生產調度甘特圖",
        "en": "Production Scheduling Gantt Chart"
    },
    "gantt_xaxis_title": {
        "zh-CN": "时间 (分钟)",
        "zh-TW": "時間 (分鐘)",
        "en": "Time (minutes)"
    },
    "gantt_yaxis_title": {
        "zh-CN": "工作站",
        "zh-TW": "工作站",
        "en": "Workstation"
    },
    "workstation": {
        "zh-CN": "工作站",
        "zh-TW": "工作站",
        "en": "Workstation"
    },
    "part_id": {
        "zh-CN": "零件ID",
        "zh-TW": "零件ID",
        "en": "Part ID"
    },
    "order_id": {
        "zh-CN": "订单ID",
        "zh-TW": "訂單ID",
        "en": "Order ID"
    },
    "start_time": {
        "zh-CN": "开始时间",
        "zh-TW": "開始時間",
        "en": "Start Time"
    },
    "end_time": {
        "zh-CN": "结束时间",
        "zh-TW": "結束時間",
        "en": "End Time"
    },
    "duration": {
        "zh-CN": "持续时间",
        "zh-TW": "持續時間",
        "en": "Duration"
    },
    "utilization_rate": {
        "zh-CN": "利用率",
        "zh-TW": "利用率",
        "en": "Utilization"
    },
    "util_chart_title": {
        "zh-CN": "各工作站设备利用率",
        "zh-TW": "各工作站設備利用率",
        "en": "Equipment Utilization per Workstation"
    },
    "utilization_rate_percent": {
        "zh-CN": "利用率 (%)",
        "zh-TW": "利用率 (%)",
        "en": "Utilization (%)"
    },
    
    # 进度提示
    "initializing": {
        "zh-CN": "🔄 初始化环境...",
        "zh-TW": "🔄 初始化環境...",
        "en": "🔄 Initializing environment..."
    },
    "starting_sim": {
        "zh-CN": "🚀 开始调度仿真...",
        "zh-TW": "🚀 開始調度仿真...",
        "en": "🚀 Starting simulation..."
    },
    "scheduling": {
        "zh-CN": "⚙️ 调度中... ({}/{} 步)",
        "zh-TW": "⚙️ 調度中... ({}/{} 步)",
        "en": "⚙️ Scheduling... ({}/{} steps)"
    },
    "loading_model": {
        "zh-CN": "正在加载模型...",
        "zh-TW": "正在加載模型...",
        "en": "Loading model..."
    },
    
    # 难度等级
    "easy": {
        "zh-CN": "🟢 简单",
        "zh-TW": "🟢 簡單",
        "en": "🟢 Easy"
    },
    "medium": {
        "zh-CN": "🟡 中等",
        "zh-TW": "🟡 中等",
        "en": "🟡 Medium"
    },
    "hard": {
        "zh-CN": "🟠 困难",
        "zh-TW": "🟠 困難",
        "en": "🟠 Hard"
    },
    "very_hard": {
        "zh-CN": "🔴 非常困难",
        "zh-TW": "🔴 非常困難",
        "en": "🔴 Very Hard"
    },
    "difficulty_very_high": {
        "zh-CN": "极高 ⚠️",
        "zh-TW": "極高 ⚠️",
        "en": "Very High ⚠️"
    },
    "difficulty_high": {
        "zh-CN": "高 🎯",
        "zh-TW": "高 🎯",
        "en": "High 🎯"
    },
    "difficulty_medium": {
        "zh-CN": "中等 ⚡",
        "zh-TW": "中等 ⚡",
        "en": "Medium ⚡"
    },
    "difficulty_low": {
        "zh-CN": "低 ✅",
        "zh-TW": "低 ✅",
        "en": "Low ✅"
    },
    "warn_makespan_too_long": {
        "zh-CN": "⚠️ 理论最短完工时间({:.1f}min)超过标准仿真时间({}min)，订单可能无法全部完成！",
        "zh-TW": "⚠️ 理論最短完工時間({:.1f}min)超過標準仿真時間({}min)，訂單可能無法全部完成！",
        "en": "⚠️ Theoretical makespan ({:.1f}min) exceeds standard simulation time ({}min), orders may not be fully completed!"
    },
    "suggestion_reduce_orders": {
        "zh-CN": "💡 建议：减少订单数量或延长交期时间",
        "zh-TW": "💡 建議：減少訂單數量或延長交期時間",
        "en": "💡 Suggestion: Reduce order quantity or extend due dates"
    },
    "info_high_challenge": {
        "zh-CN": "🎯 高挑战性任务：理论完工时间占仿真时间的{:.1f}%，时间非常紧张",
        "zh-TW": "🎯 高挑戰性任務：理論完工時間佔仿真時間的{:.1f}%，時間非常緊張",
        "en": "🎯 High challenge: Theoretical makespan is {:.1f}% of simulation time, schedule is very tight"
    },
    "info_medium_challenge": {
        "zh-CN": "⚡ 中等难度任务：理论完工时间占仿真时间的{:.1f}%，有一定挑战",
        "zh-TW": "⚡ 中等難度任務：理論完工時間佔仿真時間的{:.1f}%，有一定挑戰",
        "en": "⚡ Medium challenge: Theoretical makespan is {:.1f}% of simulation time, moderately challenging"
    },
    "info_low_challenge": {
        "zh-CN": "✅ 任务难度适中：理论完工时间占仿真时间的{:.1f}%",
        "zh-TW": "✅ 任務難度適中：理論完工時間佔仿真時間的{:.1f}%",
        "en": "✅ Moderate difficulty: Theoretical makespan is {:.1f}% of simulation time"
    },
    "warn_due_date_too_short": {
        "zh-CN": "⚠️ 部分订单交期过短(最短{:.0f}min)，可能导致严重延期",
        "zh-TW": "⚠️ 部分訂單交期過短(最短{:.0f}min)，可能導致嚴重延期",
        "en": "⚠️ Some orders have very short due dates (min {:.0f}min), which may cause significant delays"
    },
    "warn_avg_due_date_too_short": {
        "zh-CN": "⚠️ 平均交期({:.0f}min)短于理论完工时间({:.1f}min)，大部分订单可能延期",
        "zh-TW": "⚠️ 平均交期({:.0f}min)短於理論完工時間({:.1f}min)，大部分訂單可能延期",
        "en": "⚠️ Average due date ({:.0f}min) is shorter than theoretical makespan ({:.1f}min), most orders may be delayed"
    },
    "warn_bottleneck_overload": {
        "zh-CN": "🔍 瓶颈工作站'{}'负荷极高({:.0f}%)，可能严重影响整体进度",
        "zh-TW": "🔍 瓶頸工作站'{}'負荷極高({:.0f}%)，可能嚴重影響整體進度",
        "en": "🔍 Bottleneck workstation '{}' is under extremely high load ({:.0f}%), may severely impact overall progress"
    },
    "warn_bottleneck_high_load": {
        "zh-CN": "🔍 瓶颈工作站'{}'负荷较高({:.0f}%)，需要优化调度策略",
        "zh-TW": "🔍 瓶頸工作站'{}'負荷较高({:.0f}%)，需要優化調度策略",
        "en": "🔍 Bottleneck workstation '{}' is under high load ({:.0f}%), scheduling strategy needs optimization"
    },
    
    # 单位
    "minutes": {
        "zh-CN": " 分钟",
        "zh-TW": " 分鐘",
        "en": " minutes"
    },
    "pieces": {
        "zh-CN": "件",
        "zh-TW": "件",
        "en": " pcs"
    },
    
    # 启发式算法对比
    "comparison_options": {
        "zh-CN": "🔬 对比选项",
        "zh-TW": "🔬 對比選項",
        "en": "🔬 Comparison Options"
    },
    "compare_heuristics_checkbox": {
        "zh-CN": "同时运行启发式算法进行对比 (FIFO, EDD, SPT)",
        "zh-TW": "同時運行啟發式算法進行對比 (FIFO, EDD, SPT)",
        "en": "Run heuristic algorithms for comparison (FIFO, EDD, SPT)"
    },
    "compare_heuristics_help": {
        "zh-CN": "勾选后将自动运行启发式算法并展示对比结果",
        "zh-TW": "勾選後將自動運行啟發式算法並展示對比結果",
        "en": "Automatically run heuristic algorithms and show comparison results when checked"
    },
    "algorithm_performance_comparison": {
        "zh-CN": "📊 算法性能对比",
        "zh-TW": "📊 算法性能對比",
        "en": "📊 Algorithm Performance Comparison"
    },
    "heuristic_gantt_comparison": {
        "zh-CN": "🔬 启发式算法甘特图对比",
        "zh-TW": "🔬 啟發式算法甘特圖對比",
        "en": "🔬 Heuristic Algorithm Gantt Chart Comparison"
    },
    
    # 对比表格列名
    "algorithm": {
        "zh-CN": "算法",
        "zh-TW": "算法",
        "en": "Algorithm"
    },
    "completion_rate": {
        "zh-CN": "完成率",
        "zh-TW": "完成率",
        "en": "Completion Rate"
    },
    "completion_time": {
        "zh-CN": "完工时间",
        "zh-TW": "完工時間",
        "en": "Completion Time"
    },
    "avg_utilization": {
        "zh-CN": "平均利用率",
        "zh-TW": "平均利用率",
        "en": "Avg. Utilization"
    },
    "total_delay": {
        "zh-CN": "总延迟",
        "zh-TW": "總延遲",
        "en": "Total Delay"
    },
    "comprehensive_score": {
        "zh-CN": "综合得分",
        "zh-TW": "綜合得分",
        "en": "Comprehensive Score"
    },
    
    # 甘特图相关
    "gantt_chart_algorithm": {
        "zh-CN": "甘特图 - {}",
        "zh-TW": "甘特圖 - {}",
        "en": "Gantt Chart - {}"
    },
    "download_algorithm_gantt": {
        "zh-CN": "💾 下载 {} 甘特图",
        "zh-TW": "💾 下載 {} 甘特圖",
        "en": "💾 Download {} Gantt Chart"
    },
    "download_algorithm_gantt_html": {
        "zh-CN": "下载 {} 甘特图HTML",
        "zh-TW": "下載 {} 甘特圖HTML",
        "en": "Download {} Gantt Chart HTML"
    },
    "no_gantt_data_algorithm": {
        "zh-CN": "{}: 无甘特图数据",
        "zh-TW": "{}: 無甘特圖數據",
        "en": "{}: No Gantt chart data"
    },
    
    # 模型性能对比模块
    "model_comparison": {
        "zh-CN": "模型性能对比",
        "zh-TW": "模型性能對比",
        "en": "Model Performance Comparison"
    },
    "model_comparison_description": {
        "zh-CN": "对比多个模型在相同订单配置下的性能",
        "zh-TW": "對比多個模型在相同訂單配置下的性能",
        "en": "Compare multiple models' performance under the same order configuration"
    },
    "model_comparison_help": {
        "zh-CN": """
        **📊 模型性能对比功能**
        
        此功能用于在**完全相同**的订单配置和动态环境参数下，批量测试多个已训练模型的性能，支持控制变量实验。
        
        **使用步骤：**
        1. 先在"订单配置"区域配置好订单列表
        2. 设置好动态环境配置（设备故障、紧急插单）
        3. 在下方选择要对比的模型（至少2个）
        4. 设置对比参数（最大步数、运行次数）
        5. 点击"开始对比"按钮
        6. 查看对比结果（表格、雷达图、柱状图）
        
        **注意：** 运行次数越多，结果越稳定，但耗时也越长。
        """,
        "zh-TW": """
        **📊 模型性能對比功能**
        
        此功能用於在**完全相同**的訂單配置和動態環境參數下，批量測試多個已訓練模型的性能，支持控制變量實驗。
        
        **使用步驟：**
        1. 先在"訂單配置"區域配置好訂單列表
        2. 設置好動態環境配置（設備故障、緊急插單）
        3. 在下方選擇要對比的模型（至少2個）
        4. 設置對比參數（最大步數、運行次數）
        5. 點擊"開始對比"按鈕
        6. 查看對比結果（表格、雷達圖、柱狀圖）
        
        **注意：** 運行次數越多，結果越穩定，但耗時也越長。
        """,
        "en": """
        **📊 Model Performance Comparison**
        
        This feature allows batch testing of multiple trained models under **identical** order configurations and dynamic environment parameters, supporting controlled variable experiments.
        
        **Usage Steps:**
        1. Configure order list in "Order Configuration" section
        2. Set dynamic environment parameters (equipment failure, emergency orders)
        3. Select models to compare (at least 2) below
        4. Configure comparison parameters (max steps, runs)
        5. Click "Start Comparison" button
        6. View comparison results (table, radar chart, bar chart)
        
        **Note:** More runs lead to more stable results, but take longer time.
        """
    },
    "config_orders_first_comparison": {
        "zh-CN": "请先在上方配置订单后再使用模型对比功能",
        "zh-TW": "請先在上方配置訂單後再使用模型對比功能",
        "en": "Please configure orders first before using model comparison"
    },
    "current_orders_count": {
        "zh-CN": "当前订单数：{} 个",
        "zh-TW": "當前訂單數：{} 個",
        "en": "Current Orders: {} items"
    },
    "enabled": {
        "zh-CN": "已启用",
        "zh-TW": "已啟用",
        "en": "Enabled"
    },
    "disabled": {
        "zh-CN": "未启用",
        "zh-TW": "未啟用",
        "en": "Disabled"
    },
    "equipment_failure": {
        "zh-CN": "设备故障模拟",
        "zh-TW": "設備故障模擬",
        "en": "Equipment Failure Simulation"
    },
    "emergency_orders": {
        "zh-CN": "紧急插单模拟",
        "zh-TW": "緊急插單模擬",
        "en": "Emergency Orders Simulation"
    },
    "select_models_to_compare": {
        "zh-CN": "选择要对比的模型",
        "zh-TW": "選擇要對比的模型",
        "en": "Select Models to Compare"
    },
    "select_models": {
        "zh-CN": "选择模型（可多选）",
        "zh-TW": "選擇模型（可多選）",
        "en": "Select Models (Multiple)"
    },
    "select_models_help": {
        "zh-CN": "至少选择2个模型进行对比",
        "zh-TW": "至少選擇2個模型進行對比",
        "en": "Select at least 2 models to compare"
    },
    "select_at_least_two_models": {
        "zh-CN": "请至少选择2个模型进行对比",
        "zh-TW": "請至少選擇2個模型進行對比",
        "en": "Please select at least 2 models to compare"
    },
    "selected_models_count": {
        "zh-CN": "已选择 {} 个模型",
        "zh-TW": "已選擇 {} 個模型",
        "en": "{} models selected"
    },
    "max_steps": {
        "zh-CN": "最大仿真步数",
        "zh-TW": "最大仿真步數",
        "en": "Max Simulation Steps"
    },
    "max_steps_help": {
        "zh-CN": "每次仿真的最大步数，步数越多耗时越长",
        "zh-TW": "每次仿真的最大步數，步數越多耗時越長",
        "en": "Max steps per simulation, more steps take longer time"
    },
    "max_steps_comparison_help": {
        "zh-CN": "仿真环境运行的最大步数上限。模型会持续决策直到任务完成或达到此上限。可根据订单复杂度调整：简单订单500-1000步，复杂订单1500-3000步",
        "zh-TW": "仿真環境運行的最大步數上限。模型會持續決策直到任務完成或達到此上限。可根據訂單複雜度調整：簡單訂單500-1000步，複雜訂單1500-3000步",
        "en": "Max step limit for simulation environment. Model will keep making decisions until tasks complete or this limit is reached. Adjust based on order complexity: 500-1000 for simple orders, 1500-3000 for complex orders"
    },
    "select_models_instruction": {
        "zh-CN": "请勾选要对比的模型（建议2-5个）",
        "zh-TW": "請勾選要對比的模型（建議2-5個）",
        "en": "Check models to compare (2-5 recommended)"
    },
    "view_selected_models": {
        "zh-CN": "查看已选择的模型",
        "zh-TW": "查看已選擇的模型",
        "en": "View Selected Models"
    },
    "selected_models_list": {
        "zh-CN": "已选择的模型列表：",
        "zh-TW": "已選擇的模型列表：",
        "en": "Selected Models:"
    },
    "comparison_parameters": {
        "zh-CN": "对比参数设置",
        "zh-TW": "對比參數設置",
        "en": "Comparison Parameters"
    },
    "comparison_runs": {
        "zh-CN": "运行次数",
        "zh-TW": "運行次數",
        "en": "Number of Runs"
    },
    "comparison_runs_help": {
        "zh-CN": "每个模型运行的次数，用于获得平均性能（1-5次）",
        "zh-TW": "每個模型運行的次數，用於獲得平均性能（1-5次）",
        "en": "Number of runs per model to get average performance (1-5 runs)"
    },
    "start_comparison": {
        "zh-CN": "🚀 开始对比",
        "zh-TW": "🚀 開始對比",
        "en": "🚀 Start Comparison"
    },
    "running_model": {
        "zh-CN": "正在运行模型 {} (第 {}/{} 次)...",
        "zh-TW": "正在運行模型 {} (第 {}/{} 次)...",
        "en": "Running model {} (Run {}/{})..."
    },
    "load_model_failed": {
        "zh-CN": "加载模型 {} 失败",
        "zh-TW": "加載模型 {} 失敗",
        "en": "Failed to load model {}"
    },
    "scheduling_failed": {
        "zh-CN": "模型 {} 调度失败",
        "zh-TW": "模型 {} 調度失敗",
        "en": "Model {} scheduling failed"
    },
    "comparison_completed": {
        "zh-CN": "✅ 对比完成！结果如下：",
        "zh-TW": "✅ 對比完成！結果如下：",
        "en": "✅ Comparison completed! Results:"
    },
    "comparison_failed": {
        "zh-CN": "❌ 对比失败，请检查模型和订单配置",
        "zh-TW": "❌ 對比失敗，請檢查模型和訂單配置",
        "en": "❌ Comparison failed, please check models and order configuration"
    },
    "comparison_results": {
        "zh-CN": "📊 对比结果",
        "zh-TW": "📊 對比結果",
        "en": "📊 Comparison Results"
    },
    "model_name": {
        "zh-CN": "模型名称",
        "zh-TW": "模型名稱",
        "en": "Model Name"
    },
    "avg_makespan": {
        "zh-CN": "平均完工时间",
        "zh-TW": "平均完工時間",
        "en": "Avg Makespan"
    },
    "avg_utilization": {
        "zh-CN": "平均利用率",
        "zh-TW": "平均利用率",
        "en": "Avg Utilization"
    },
    "avg_tardiness": {
        "zh-CN": "平均延迟",
        "zh-TW": "平均延遲",
        "en": "Avg Tardiness"
    },
    "avg_score": {
        "zh-CN": "平均评分",
        "zh-TW": "平均評分",
        "en": "Avg Score"
    },
    "avg_reward": {
        "zh-CN": "平均奖励",
        "zh-TW": "平均獎勵",
        "en": "Avg Reward"
    },
    "runs": {
        "zh-CN": "运行次数",
        "zh-TW": "運行次數",
        "en": "Runs"
    },
    "radar_chart_comparison": {
        "zh-CN": "📊 雷达图对比",
        "zh-TW": "📊 雷達圖對比",
        "en": "📊 Radar Chart Comparison"
    },
    "utilization": {
        "zh-CN": "利用率",
        "zh-TW": "利用率",
        "en": "Utilization"
    },
    "score": {
        "zh-CN": "评分",
        "zh-TW": "評分",
        "en": "Score"
    },
    "model_performance_radar": {
        "zh-CN": "模型性能雷达图对比",
        "zh-TW": "模型性能雷達圖對比",
        "en": "Model Performance Radar Chart"
    },
    "bar_chart_comparison": {
        "zh-CN": "📊 完工率对比",
        "zh-TW": "📊 完工率對比",
        "en": "📊 Completion Rate Comparison"
    },
    "completion_rate_comparison": {
        "zh-CN": "模型完工率对比",
        "zh-TW": "模型完工率對比",
        "en": "Model Completion Rate Comparison"
    },
    "clear_comparison_results": {
        "zh-CN": "🗑️ 清除对比结果",
        "zh-TW": "🗑️ 清除對比結果",
        "en": "🗑️ Clear Comparison Results"
    },
}

def get_text(key: str, lang: str = "zh-CN", *args) -> str:
    """
    获取指定语言的文本
    
    Args:
        key: 文本键
        lang: 语言代码 ("zh-CN", "zh-TW", "en")
        *args: 格式化参数
    
    Returns:
        翻译后的文本
    """
    if key not in TEXTS:
        return key
    
    text = TEXTS[key].get(lang, TEXTS[key].get("zh-CN", key))
    
    # 如果有格式化参数，进行格式化
    if args:
        try:
            text = text.format(*args)
        except:
            pass
    
    return text

