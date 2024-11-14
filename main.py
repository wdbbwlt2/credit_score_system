# main.py - Part 1

import os
import sys
import json
import tkinter as tk
from datetime import datetime
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
import logging
from datetime import datetime
import pandas as pd
from typing import Dict, List, Optional
import threading
from sqlalchemy import create_engine, text

from credit_score_system.core import CreditScoreSystem  # 添加这行
from credit_score_system.utils import (
    DatabaseConfig,
    DatabaseManager,
    DataLoader,
    ConfigLoader,
    ScoreLogger,
    CreditScoreUtils,  # 添加这一行
    CreditScoreException,
    DataValidationError,
    ScoreCalculationError,
    DatabaseError,
    ConfigurationError
)

# 设置日志
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f"main_{datetime.now().strftime('%Y%m%d')}.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class CreditScoreApp:
    def __init__(self, root):
        """初始化信用评分系统应用"""
        self.root = root
        self.root.title("客户信用评分系统")
        self.root.geometry("800x600")

        # 初始化状态变量
        self.current_year = datetime.now().year
        self.current_month = datetime.now().month

        # 初始化系统
        self.initialize_system()

        # 创建主界面
        self.create_gui()

    def initialize_system(self):
        """初始化评分系统"""
        try:
            config_path = os.path.join(os.path.dirname(__file__), 'config.json')
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            self.system = CreditScoreSystem(config)
            logger.info("系统初始化成功")

        except Exception as e:
            logger.error(f"系统初始化失败: {str(e)}")
            messagebox.showerror("错误", f"系统初始化失败: {str(e)}")
            sys.exit(1)

    def create_gui(self):
        """创建图形界面"""
        # 创建notebook（选项卡）
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill='both', padx=5, pady=5)

        # 创建各个功能页面
        self.create_single_score_tab()
        self.create_batch_score_tab()
        self.create_query_tab()
        self.create_statistics_tab()
        self.create_settings_tab()

        # 创建状态栏
        self.status_bar = tk.Label(
            self.root,
            text="就绪",
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def create_single_score_tab(self):
        """创建单客户评分页面"""
        single_frame = ttk.Frame(self.notebook)
        self.notebook.add(single_frame, text="单客户评分")

        # 客户信息输入区
        input_frame = ttk.LabelFrame(single_frame, text="客户信息")
        input_frame.pack(fill='x', padx=5, pady=5)

        ttk.Label(input_frame, text="客户编号:").grid(row=0, column=0, padx=5, pady=5)
        self.customer_code_var = tk.StringVar()
        self.customer_code_entry = ttk.Entry(input_frame, textvariable=self.customer_code_var)
        self.customer_code_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="年份:").grid(row=1, column=0, padx=5, pady=5)
        self.year_var = tk.StringVar(value=str(self.current_year))
        self.year_entry = ttk.Entry(input_frame, textvariable=self.year_var)
        self.year_entry.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="月份:").grid(row=2, column=0, padx=5, pady=5)
        self.month_var = tk.StringVar(value=str(self.current_month))
        self.month_entry = ttk.Entry(input_frame, textvariable=self.month_var)
        self.month_entry.grid(row=2, column=1, padx=5, pady=5)

        # 按钮
        btn_frame = ttk.Frame(single_frame)
        btn_frame.pack(fill='x', padx=5, pady=5)

        ttk.Button(
            btn_frame,
            text="计算评分",
            command=self.calculate_single_score
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            btn_frame,
            text="清除",
            command=self.clear_single_score
        ).pack(side=tk.LEFT, padx=5)

        # 结果显示区
        result_frame = ttk.LabelFrame(single_frame, text="评分结果")
        result_frame.pack(fill='both', expand=True, padx=5, pady=5)

        self.result_text = tk.Text(result_frame, height=15)
        self.result_text.pack(fill='both', expand=True, padx=5, pady=5)

    # main.py - Part 2

    def create_batch_score_tab(self):
        """创建批量评分页面"""
        batch_frame = ttk.Frame(self.notebook)
        self.notebook.add(batch_frame, text="批量评分")

        # 批量评分控制区
        control_frame = ttk.LabelFrame(batch_frame, text="评分控制")
        control_frame.pack(fill='x', padx=5, pady=5)

        # 时间范围选择
        time_frame = ttk.Frame(control_frame)
        time_frame.pack(fill='x', padx=5, pady=5)

        ttk.Label(time_frame, text="起始年月:").grid(row=0, column=0, padx=5, pady=5)
        self.start_year_var = tk.StringVar(value=str(self.current_year))
        self.start_year_entry = ttk.Entry(time_frame, textvariable=self.start_year_var, width=6)
        self.start_year_entry.grid(row=0, column=1, padx=5, pady=5)

        self.start_month_var = tk.StringVar(value="1")
        self.start_month_entry = ttk.Entry(time_frame, textvariable=self.start_month_var, width=4)
        self.start_month_entry.grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(time_frame, text="结束年月:").grid(row=0, column=3, padx=5, pady=5)
        self.end_year_var = tk.StringVar(value=str(self.current_year))
        self.end_year_entry = ttk.Entry(time_frame, textvariable=self.end_year_var, width=6)
        self.end_year_entry.grid(row=0, column=4, padx=5, pady=5)

        self.end_month_var = tk.StringVar(value=str(self.current_month))
        self.end_month_entry = ttk.Entry(time_frame, textvariable=self.end_month_var, width=4)
        self.end_month_entry.grid(row=0, column=5, padx=5, pady=5)

        # 客户选择
        customer_frame = ttk.Frame(control_frame)
        customer_frame.pack(fill='x', padx=5, pady=5)

        ttk.Label(customer_frame, text="客户列表(可选):").pack(side=tk.LEFT, padx=5)
        self.customer_list_var = tk.StringVar()
        self.customer_list_entry = ttk.Entry(customer_frame, textvariable=self.customer_list_var, width=40)
        self.customer_list_entry.pack(side=tk.LEFT, padx=5)

        ttk.Button(
            customer_frame,
            text="导入客户",
            command=self.import_customer_list
        ).pack(side=tk.LEFT, padx=5)

        # 控制按钮
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill='x', padx=5, pady=5)

        ttk.Button(
            btn_frame,
            text="开始批量评分",
            command=self.start_batch_scoring
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            btn_frame,
            text="停止",
            command=self.stop_batch_scoring
        ).pack(side=tk.LEFT, padx=5)

        # 进度显示
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            control_frame,
            variable=self.progress_var,
            maximum=100
        )
        self.progress_bar.pack(fill='x', padx=5, pady=5)

        # 结果显示区
        result_frame = ttk.LabelFrame(batch_frame, text="评分结果")
        result_frame.pack(fill='both', expand=True, padx=5, pady=5)

        self.batch_result_text = tk.Text(result_frame)
        self.batch_result_text.pack(fill='both', expand=True, padx=5, pady=5)

    def create_query_tab(self):
        """创建查询页面"""
        query_frame = ttk.Frame(self.notebook)
        self.notebook.add(query_frame, text="评分查询")

        # 查询条件区
        search_frame = ttk.LabelFrame(query_frame, text="查询条件")
        search_frame.pack(fill='x', padx=5, pady=5)

        # 客户编号
        code_frame = ttk.Frame(search_frame)
        code_frame.pack(fill='x', padx=5, pady=5)

        ttk.Label(code_frame, text="客户编号:").pack(side=tk.LEFT, padx=5)
        self.search_code_var = tk.StringVar()
        self.search_code_entry = ttk.Entry(code_frame, textvariable=self.search_code_var)
        self.search_code_entry.pack(side=tk.LEFT, padx=5)

        # 评分范围
        score_frame = ttk.Frame(search_frame)
        score_frame.pack(fill='x', padx=5, pady=5)

        ttk.Label(score_frame, text="评分范围:").pack(side=tk.LEFT, padx=5)
        self.min_score_var = tk.StringVar()
        ttk.Entry(score_frame, textvariable=self.min_score_var, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Label(score_frame, text="至").pack(side=tk.LEFT)
        self.max_score_var = tk.StringVar()
        ttk.Entry(score_frame, textvariable=self.max_score_var, width=5).pack(side=tk.LEFT, padx=5)

        # 时间范围
        time_frame = ttk.Frame(search_frame)
        time_frame.pack(fill='x', padx=5, pady=5)

        ttk.Label(time_frame, text="评分时间:").pack(side=tk.LEFT, padx=5)
        self.search_start_date_var = tk.StringVar()
        ttk.Entry(time_frame, textvariable=self.search_start_date_var, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(time_frame, text="至").pack(side=tk.LEFT)
        self.search_end_date_var = tk.StringVar()
        ttk.Entry(time_frame, textvariable=self.search_end_date_var, width=10).pack(side=tk.LEFT, padx=5)

        # 查询按钮
        btn_frame = ttk.Frame(search_frame)
        btn_frame.pack(fill='x', padx=5, pady=5)

        ttk.Button(
            btn_frame,
            text="查询",
            command=self.search_scores
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            btn_frame,
            text="导出结果",
            command=self.export_search_results
        ).pack(side=tk.LEFT, padx=5)

        # 结果显示区
        result_frame = ttk.LabelFrame(query_frame, text="查询结果")
        result_frame.pack(fill='both', expand=True, padx=5, pady=5)

        # 使用Treeview显示查询结果
        self.result_tree = ttk.Treeview(
            result_frame,
            columns=("客户编号", "客户名称", "评分年月", "最终得分", "信用等级"),
            show="headings"
        )

        # 设置列标题
        for col in self.result_tree["columns"]:
            self.result_tree.heading(col, text=col)
            self.result_tree.column(col, width=100)

        # 添加滚动条
        scrollbar = ttk.Scrollbar(result_frame, orient="vertical", command=self.result_tree.yview)
        self.result_tree.configure(yscrollcommand=scrollbar.set)

        self.result_tree.pack(side=tk.LEFT, fill='both', expand=True)
        scrollbar.pack(side=tk.RIGHT, fill='y')

    # main.py - Part 3

    def create_statistics_tab(self):
        """创建统计分析页面"""
        stats_frame = ttk.Frame(self.notebook)
        self.notebook.add(stats_frame, text="统计分析")

        # 统计控制区
        control_frame = ttk.LabelFrame(stats_frame, text="统计选项")
        control_frame.pack(fill='x', padx=5, pady=5)

        # 统计类型选择
        type_frame = ttk.Frame(control_frame)
        type_frame.pack(fill='x', padx=5, pady=5)

        ttk.Label(type_frame, text="统计类型:").pack(side=tk.LEFT, padx=5)
        self.stats_type_var = tk.StringVar(value="评分分布")
        stats_type_combo = ttk.Combobox(
            type_frame,
            textvariable=self.stats_type_var,
            values=["评分分布", "客户类型分析", "月度趋势", "评分变化"]
        )
        stats_type_combo.pack(side=tk.LEFT, padx=5)

        # 时间范围
        time_frame = ttk.Frame(control_frame)
        time_frame.pack(fill='x', padx=5, pady=5)

        ttk.Label(time_frame, text="统计期间:").pack(side=tk.LEFT, padx=5)
        self.stats_start_date_var = tk.StringVar(value=f"{self.current_year}-01")
        ttk.Entry(time_frame, textvariable=self.stats_start_date_var, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Label(time_frame, text="至").pack(side=tk.LEFT)
        self.stats_end_date_var = tk.StringVar(value=f"{self.current_year}-{self.current_month:02d}")
        ttk.Entry(time_frame, textvariable=self.stats_end_date_var, width=8).pack(side=tk.LEFT, padx=5)

        # 控制按钮
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill='x', padx=5, pady=5)

        ttk.Button(
            btn_frame,
            text="生成统计",
            command=self.generate_statistics
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            btn_frame,
            text="导出报告",
            command=self.export_statistics
        ).pack(side=tk.LEFT, padx=5)

        # 统计结果显示区
        result_frame = ttk.LabelFrame(stats_frame, text="统计结果")
        result_frame.pack(fill='both', expand=True, padx=5, pady=5)

        # 创建左右分栏
        paned = ttk.PanedWindow(result_frame, orient=tk.HORIZONTAL)
        paned.pack(fill='both', expand=True)

        # 左侧图表区域
        chart_frame = ttk.Frame(paned)
        self.chart_canvas = tk.Canvas(chart_frame)
        self.chart_canvas.pack(fill='both', expand=True)
        paned.add(chart_frame)

        # 右侧数据明细
        detail_frame = ttk.Frame(paned)
        self.stats_detail_text = tk.Text(detail_frame, width=40)
        self.stats_detail_text.pack(fill='both', expand=True)
        paned.add(detail_frame)

    def create_settings_tab(self):
        """创建设置页面"""
        settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(settings_frame, text="系统设置")

        # 数据库设置
        db_frame = ttk.LabelFrame(settings_frame, text="数据库设置")
        db_frame.pack(fill='x', padx=5, pady=5)

        # 连接信息
        conn_frame = ttk.Frame(db_frame)
        conn_frame.pack(fill='x', padx=5, pady=5)

        ttk.Label(conn_frame, text="主机:").grid(row=0, column=0, padx=5, pady=5)
        self.db_host_var = tk.StringVar(value=self.system.db_manager.config.host)
        ttk.Entry(conn_frame, textvariable=self.db_host_var).grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(conn_frame, text="端口:").grid(row=0, column=2, padx=5, pady=5)
        self.db_port_var = tk.StringVar(value=str(self.system.db_manager.config.port))
        ttk.Entry(conn_frame, textvariable=self.db_port_var, width=6).grid(row=0, column=3, padx=5, pady=5)

        ttk.Label(conn_frame, text="数据库:").grid(row=1, column=0, padx=5, pady=5)
        self.db_name_var = tk.StringVar(value=self.system.db_manager.config.database)
        ttk.Entry(conn_frame, textvariable=self.db_name_var).grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(conn_frame, text="用户名:").grid(row=2, column=0, padx=5, pady=5)
        self.db_user_var = tk.StringVar(value=self.system.db_manager.config.user)
        ttk.Entry(conn_frame, textvariable=self.db_user_var).grid(row=2, column=1, padx=5, pady=5)

        ttk.Label(conn_frame, text="密码:").grid(row=2, column=2, padx=5, pady=5)
        self.db_pass_var = tk.StringVar(value=self.system.db_manager.config.password)
        ttk.Entry(conn_frame, textvariable=self.db_pass_var, show="*").grid(row=2, column=3, padx=5, pady=5)

        # 测试连接按钮
        ttk.Button(
            db_frame,
            text="测试连接",
            command=self.test_db_connection
        ).pack(padx=5, pady=5)

        # 评分设置
        score_frame = ttk.LabelFrame(settings_frame, text="评分设置")
        score_frame.pack(fill='x', padx=5, pady=5)

        # 权重设置
        weight_frame = ttk.Frame(score_frame)
        weight_frame.pack(fill='x', padx=5, pady=5)

        ttk.Label(weight_frame, text="基础评分权重:").grid(row=0, column=0, padx=5, pady=5)
        self.base_weight_var = tk.StringVar(value="35")
        ttk.Entry(weight_frame, textvariable=self.base_weight_var, width=5).grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(weight_frame, text="交易评分权重:").grid(row=1, column=0, padx=5, pady=5)
        self.trans_weight_var = tk.StringVar(value="35")
        ttk.Entry(weight_frame, textvariable=self.trans_weight_var, width=5).grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(weight_frame, text="履约评分权重:").grid(row=2, column=0, padx=5, pady=5)
        self.contract_weight_var = tk.StringVar(value="30")
        ttk.Entry(weight_frame, textvariable=self.contract_weight_var, width=5).grid(row=2, column=1, padx=5, pady=5)

        # 保存设置
        ttk.Button(
            settings_frame,
            text="保存设置",
            command=self.save_settings
        ).pack(padx=5, pady=10)

    # main.py - Part 4

    def calculate_single_score(self):
        """计算单个客户评分"""
        try:
            customer_code = self.customer_code_var.get().strip()
            year = int(self.year_var.get())
            month = int(self.month_var.get())

            if not customer_code:
                messagebox.showwarning("警告", "请输入客户编号")
                return

            # 更新状态
            self.status_bar['text'] = f"正在计算客户 {customer_code} 的评分..."
            self.root.update()

            # 加载数据
            sales_df, collection_df, contract_df = self.system.data_loader.load_monthly_data(
                year, month
            )

            # 计算评分
            result = self.system.calculator.process_and_verify(
                customer_code=customer_code,
                year=year,
                month=month,
                sales_df=sales_df,
                collection_df=collection_df,
                contract_df=contract_df
            )

            # 显示结果
            self.display_single_score_result(result)

            # 更新状态
            self.status_bar['text'] = "评分计算完成"

        except Exception as e:
            logger.error(f"计算评分时发生错误: {str(e)}")
            messagebox.showerror("错误", f"计算评分失败: {str(e)}")
            self.status_bar['text'] = "评分计算失败"

    def display_single_score_result(self, result: Dict):
        """显示单个客户评分结果"""
        self.result_text.delete(1.0, tk.END)

        if not result or not result.get('score_details'):
            self.result_text.insert(tk.END, "计算失败，未获取到评分结果\n")
            return

        score_details = result['score_details']
        validation_result = result.get('validation_result', {})

        # 显示评分详情
        self.result_text.insert(tk.END, "=== 评分详情 ===\n\n")
        self.result_text.insert(tk.END, f"基础评分: {score_details['base_score']:.2f}\n")
        self.result_text.insert(tk.END, f"交易评分: {score_details['transaction_score']:.2f}\n")
        self.result_text.insert(tk.END, f"履约评分: {score_details['contract_score']:.2f}\n")
        self.result_text.insert(tk.END, f"调整系数: {score_details['adjustment_factor']:.2f}\n")
        self.result_text.insert(tk.END, f"最终得分: {score_details['final_score']:.2f}\n")
        self.result_text.insert(tk.END, f"信用等级: {score_details['credit_level']}\n\n")

        # 显示验证结果
        self.result_text.insert(tk.END, "=== 验证结果 ===\n\n")
        self.result_text.insert(tk.END, f"验证状态: {'通过' if validation_result.get('success', False) else '失败'}\n")

        if validation_result.get('errors'):
            self.result_text.insert(tk.END, "\n错误信息:\n")
            for error in validation_result['errors']:
                self.result_text.insert(tk.END, f"- {error}\n")

        if validation_result.get('warnings'):
            self.result_text.insert(tk.END, "\n警告信息:\n")
            for warning in validation_result['warnings']:
                self.result_text.insert(tk.END, f"- {warning}\n")

    def clear_single_score(self):
        """清除单个客户评分结果"""
        self.customer_code_var.set("")
        self.year_var.set(str(self.current_year))
        self.month_var.set(str(self.current_month))
        self.result_text.delete(1.0, tk.END)

    def import_customer_list(self):
        """导入客户列表"""
        file_path = filedialog.askopenfilename(
            title="选择客户列表文件",
            filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")]
        )

        if not file_path:
            return

        try:
            # 读取文件
            with open(file_path, 'r', encoding='utf-8') as f:
                customers = [line.strip() for line in f if line.strip()]

            # 更新客户列表
            self.customer_list_var.set(", ".join(customers))
            messagebox.showinfo("成功", f"已导入 {len(customers)} 个客户")

        except Exception as e:
            logger.error(f"导入客户列表失败: {str(e)}")
            messagebox.showerror("错误", f"导入客户列表失败: {str(e)}")

    def start_batch_scoring(self):
        """开始批量评分"""
        try:
            # 获取参数
            start_year = int(self.start_year_var.get())
            start_month = int(self.start_month_var.get())
            end_year = int(self.end_year_var.get())
            end_month = int(self.end_month_var.get())

            # 获取客户列表
            customer_list = self.customer_list_var.get().strip()
            if customer_list:
                customers = [c.strip() for c in customer_list.split(",")]
            else:
                # 如果没有指定客户，获取所有活跃客户
                customers = self.system.data_loader.get_active_customers(end_year, end_month)

            # 创建批量评分线程
            self.batch_thread = threading.Thread(
                target=self._batch_scoring_worker,
                args=(start_year, start_month, end_year, end_month, customers)
            )
            self.batch_thread.daemon = True
            self.batch_thread.start()

            # 更新UI状态
            self.status_bar['text'] = "正在进行批量评分..."

        except Exception as e:
            logger.error(f"启动批量评分失败: {str(e)}")
            messagebox.showerror("错误", f"启动批量评分失败: {str(e)}")

    # main.py - Part 5

    def _batch_scoring_worker(self, start_year: int, start_month: int,
                              end_year: int, end_month: int, customers: List[str]):
        """批量评分工作线程"""
        try:
            total_tasks = len(customers)
            completed = 0

            self.batch_result_text.delete(1.0, tk.END)
            self.batch_result_text.insert(tk.END, f"开始批量评分，共 {total_tasks} 个客户\n\n")

            success_count = 0
            fail_count = 0

            for year in range(start_year, end_year + 1):
                for month in range(1, 13):
                    if year == start_year and month < start_month:
                        continue
                    if year == end_year and month > end_month:
                        break

                    # 添加这行来格式化年月
                    year_month = CreditScoreUtils.format_year_month(year, month)

                    try:
                        # 加载月度数据
                        sales_df, collection_df, contract_df = self.system.data_loader.load_monthly_data(
                            year, month
                        )

                        for customer_code in customers:
                            try:
                                # 计算评分
                                result = self.system.calculator.process_and_verify(
                                    customer_code=customer_code,
                                    year=year,
                                    month=month,
                                    sales_df=sales_df,
                                    collection_df=collection_df,
                                    contract_df=contract_df
                                )

                                # 更新结果
                                if result['validation_result']['success']:
                                    success_count += 1
                                    self.batch_result_text.insert(tk.END,
                                                                  f"✓ {customer_code} ({year_month}): "
                                                                  f"{result['score_details']['final_score']:.2f} "
                                                                  f"[{result['score_details']['credit_level']}]\n")
                                else:
                                    fail_count += 1
                                    self.batch_result_text.insert(tk.END,
                                                                  f"✗ {customer_code} ({year_month}): 计算失败\n")

                                completed += 1
                                progress = (completed / total_tasks) * 100
                                self.progress_var.set(progress)

                                # 更新UI
                                self.root.update()

                            except Exception as e:
                                fail_count += 1
                                logger.error(f"处理客户 {customer_code} 时发生错误: {str(e)}")
                                self.batch_result_text.insert(tk.END,
                                                              f"✗ {customer_code} ({year_month}): {str(e)}\n")

                    except Exception as e:
                        logger.error(f"处理 {year_month} 数据时发生错误: {str(e)}")
                        self.batch_result_text.insert(tk.END, f"加载 {year_month} 数据失败: {str(e)}\n")
                        continue

            # 显示统计结果
            self.batch_result_text.insert(tk.END, f"\n批量评分完成:\n")
            self.batch_result_text.insert(tk.END, f"成功: {success_count}\n")
            self.batch_result_text.insert(tk.END, f"失败: {fail_count}\n")
            self.status_bar['text'] = "批量评分完成"

        except Exception as e:
            logger.error(f"批量评分过程发生错误: {str(e)}")
            self.batch_result_text.insert(tk.END, f"\n批量评分过程发生错误: {str(e)}\n")
            self.status_bar['text'] = "批量评分失败"

        finally:
            self.progress_var.set(0)

    def stop_batch_scoring(self):
        """停止批量评分"""
        if hasattr(self, 'batch_thread') and self.batch_thread.is_alive():
            # 设置停止标志
            self.stop_flag = True
            self.batch_thread.join()
            self.status_bar['text'] = "批量评分已停止"
            messagebox.showinfo("提示", "批量评分已停止")

    def search_scores(self):
        """查询评分记录"""
        try:
            # 构建查询条件
            conditions = []
            params = {}

            # 客户编号
            customer_code = self.search_code_var.get().strip()
            if customer_code:
                conditions.append("customer_code = :customer_code")
                params['customer_code'] = customer_code

            # 评分范围
            min_score = self.min_score_var.get().strip()
            if min_score:
                conditions.append("final_score >= :min_score")
                params['min_score'] = float(min_score)

            max_score = self.max_score_var.get().strip()
            if max_score:
                conditions.append("final_score <= :max_score")
                params['max_score'] = float(max_score)

            # 时间范围
            start_date = self.search_start_date_var.get().strip()
            if start_date:
                year, month = map(int, start_date.split('-'))
                conditions.append(
                    "(score_year > :start_year OR (score_year = :start_year AND score_month >= :start_month))")
                params.update({
                    'start_year': year,
                    'start_month': month
                })

            end_date = self.search_end_date_var.get().strip()
            if end_date:
                year, month = map(int, end_date.split('-'))
                conditions.append("(score_year < :end_year OR (score_year = :end_year AND score_month <= :end_month))")
                params.update({
                    'end_year': year,
                    'end_month': month
                })

            # 构建查询语句
            query = """
                SELECT 
                    customer_code,
                    customer_name,
                    score_year,
                    score_month,
                    final_score,
                    credit_level
                FROM customer_credit_score
            """

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY score_year DESC, score_month DESC, customer_code"

            # 执行查询
            with self.system.db_manager.get_connection() as conn:
                results = conn.execute(text(query), params).fetchall()

            # 显示结果
            self.display_search_results(results)

        except Exception as e:
            logger.error(f"查询评分记录失败: {str(e)}")
            messagebox.showerror("错误", f"查询评分记录失败: {str(e)}")

    def display_search_results(self, results):
        """显示查询结果"""
        # 清空现有数据
        for item in self.result_tree.get_children():
            self.result_tree.delete(item)

        # 添加新数据
        for row in results:
            self.result_tree.insert("", "end", values=(
                row.customer_code,
                row.customer_name,
                f"{row.score_year}-{row.score_month:02d}",
                f"{row.final_score:.2f}",
                row.credit_level
            ))

        self.status_bar['text'] = f"查询完成，共找到 {len(results)} 条记录"

    # main.py - Part 6

    def generate_statistics(self):
        """生成统计分析"""
        try:
            stats_type = self.stats_type_var.get()
            start_date = self.stats_start_date_var.get()
            end_date = self.stats_end_date_var.get()

            self.status_bar['text'] = f"正在生成{stats_type}统计..."
            self.root.update()

            if stats_type == "评分分布":
                self._generate_score_distribution(start_date, end_date)
            elif stats_type == "客户类型分析":
                self._generate_customer_type_analysis(start_date, end_date)
            elif stats_type == "月度趋势":
                self._generate_monthly_trend(start_date, end_date)
            elif stats_type == "评分变化":
                self._generate_score_changes(start_date, end_date)

            self.status_bar['text'] = "统计分析生成完成"

        except Exception as e:
            logger.error(f"生成统计分析失败: {str(e)}")
            messagebox.showerror("错误", f"生成统计分析失败: {str(e)}")

    def _generate_score_distribution(self, start_date: str, end_date: str):
        """生成评分分布统计"""
        query = """
            SELECT 
                CASE
                    WHEN final_score >= 90 THEN 'AAA级 (90-100)'
                    WHEN final_score >= 80 THEN 'AA级 (80-89)'
                    WHEN final_score >= 70 THEN 'A级 (70-79)'
                    WHEN final_score >= 60 THEN 'BBB级 (60-69)'
                    WHEN final_score >= 50 THEN 'BB级 (50-59)'
                    WHEN final_score >= 40 THEN 'B级 (40-49)'
                    ELSE 'C级 (0-39)'
                END as score_range,
                COUNT(*) as count,
                AVG(final_score) as avg_score
            FROM customer_credit_score
            WHERE CONCAT(score_year, '-', LPAD(score_month, 2, '0')) 
                BETWEEN :start_date AND :end_date
            GROUP BY 
                CASE
                    WHEN final_score >= 90 THEN 'AAA级 (90-100)'
                    WHEN final_score >= 80 THEN 'AA级 (80-89)'
                    WHEN final_score >= 70 THEN 'A级 (70-79)'
                    WHEN final_score >= 60 THEN 'BBB级 (60-69)'
                    WHEN final_score >= 50 THEN 'BB级 (50-59)'
                    WHEN final_score >= 40 THEN 'B级 (40-49)'
                    ELSE 'C级 (0-39)'
                END
            ORDER BY MIN(final_score)
        """

        try:
            with self.system.db_manager.get_connection() as conn:
                df = pd.read_sql(query, conn, params={
                    'start_date': start_date,
                    'end_date': end_date
                })

            # 绘制图表
            self.chart_canvas.delete('all')
            figure = Figure(figsize=(8, 6))
            ax = figure.add_subplot(111)

            # 绘制柱状图
            bars = ax.bar(df['score_range'], df['count'])

            # 添加数据标签
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height,
                    f'{int(height)}',
                    ha='center',
                    va='bottom'
                )

            ax.set_title('评分分布统计')
            ax.set_xlabel('评分区间')
            ax.set_ylabel('客户数量')

            # 旋转x轴标签
            plt.xticks(rotation=45)

            # 显示图表
            canvas = FigureCanvasTkAgg(figure, self.chart_canvas)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)

            # 显示详细数据
            self.stats_detail_text.delete(1.0, tk.END)
            self.stats_detail_text.insert(tk.END, "评分分布详情:\n\n")

            total_customers = df['count'].sum()
            for _, row in df.iterrows():
                percentage = (row['count'] / total_customers) * 100
                self.stats_detail_text.insert(tk.END,
                                              f"{row['score_range']}:\n"
                                              f"  客户数: {int(row['count'])}\n"
                                              f"  占比: {percentage:.2f}%\n"
                                              f"  平均分: {row['avg_score']:.2f}\n\n"
                                              )

        except Exception as e:
            logger.error(f"生成评分分布统计失败: {str(e)}")
            raise

    def test_db_connection(self):
        """测试数据库连接"""
        try:
            # 构建新的配置
            config = DatabaseConfig(
                host=self.db_host_var.get(),
                port=int(self.db_port_var.get()),
                database=self.db_name_var.get(),
                user=self.db_user_var.get(),
                password=self.db_pass_var.get()
            )

            # 创建临时连接测试
            db_manager = DatabaseManager(config)
            with db_manager.get_connection() as conn:
                conn.execute(text("SELECT 1"))

            messagebox.showinfo("成功", "数据库连接测试成功")

        except Exception as e:
            logger.error(f"数据库连接测试失败: {str(e)}")
            messagebox.showerror("错误", f"数据库连接测试失败: {str(e)}")

    def save_settings(self):
        """保存设置"""
        try:
            # 验证权重设置
            base_weight = float(self.base_weight_var.get())
            trans_weight = float(self.trans_weight_var.get())
            contract_weight = float(self.contract_weight_var.get())

            total_weight = base_weight + trans_weight + contract_weight
            if not math.isclose(total_weight, 100, rel_tol=1e-5):
                messagebox.showwarning("警告", "权重之和必须等于100")
                return

            # 保存数据库配置
            config = {
                'database': {
                    'host': self.db_host_var.get(),
                    'port': int(self.db_port_var.get()),
                    'database': self.db_name_var.get(),
                    'user': self.db_user_var.get(),
                    'password': self.db_pass_var.get()
                }
            }

            # 保存到配置文件
            config_path = os.path.join(os.path.dirname(__file__), 'config.json')
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4)

            # 保存评分权重到数据库
            self._save_score_weights(base_weight, trans_weight, contract_weight)

            messagebox.showinfo("成功", "设置保存成功")

        except Exception as e:
            logger.error(f"保存设置失败: {str(e)}")
            messagebox.showerror("错误", f"保存设置失败: {str(e)}")

    # main.py - Part 7 (Final Part)

    def export_search_results(self):
        """导出查询结果"""
        if not self.result_tree.get_children():
            messagebox.showwarning("警告", "没有可导出的数据")
            return

        try:
            # 选择保存路径
            file_path = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv")]
            )

            if not file_path:
                return

            # 获取数据
            data = []
            columns = ["客户编号", "客户名称", "评分年月", "最终得分", "信用等级"]

            for item in self.result_tree.get_children():
                values = self.result_tree.item(item)['values']
                data.append(values)

            # 创建DataFrame
            df = pd.DataFrame(data, columns=columns)

            # 导出文件
            if file_path.endswith('.xlsx'):
                df.to_excel(file_path, index=False, engine='openpyxl')
            else:
                df.to_csv(file_path, index=False, encoding='utf-8-sig')

            messagebox.showinfo("成功", "数据导出成功")

        except Exception as e:
            logger.error(f"导出数据失败: {str(e)}")
            messagebox.showerror("错误", f"导出数据失败: {str(e)}")

    def export_statistics(self):
        """导出统计报告"""
        try:
            # 选择保存路径
            file_path = filedialog.asksaveasfilename(
                defaultextension=".pdf",
                filetypes=[("PDF files", "*.pdf"), ("Word files", "*.docx")]
            )

            if not file_path:
                return

            # 获取统计数据
            stats_type = self.stats_type_var.get()
            start_date = self.stats_start_date_var.get()
            end_date = self.stats_end_date_var.get()

            # 生成报告
            self._generate_report(
                file_path,
                stats_type,
                start_date,
                end_date
            )

            messagebox.showinfo("成功", "统计报告导出成功")

        except Exception as e:
            logger.error(f"导出统计报告失败: {str(e)}")
            messagebox.showerror("错误", f"导出统计报告失败: {str(e)}")

    def _generate_report(self, file_path: str, stats_type: str,
                         start_date: str, end_date: str):
        """生成统计报告"""
        # 根据文件类型选择不同的处理方式
        if file_path.endswith('.pdf'):
            self._generate_pdf_report(file_path, stats_type, start_date, end_date)
        else:
            self._generate_word_report(file_path, stats_type, start_date, end_date)

    def _save_score_weights(self, base_weight: float, trans_weight: float,
                            contract_weight: float):
        """保存评分权重到数据库"""
        query = """
            INSERT INTO credit_score_config (
                config_type,
                config_key,
                config_value,
                effective_date
            ) VALUES (
                'weight',
                :key,
                :value,
                CURRENT_DATE
            )
        """

        try:
            with self.system.db_manager.get_connection() as conn:
                # 保存基础评分权重
                conn.execute(text(query), {
                    'key': 'base_score',
                    'value': str(base_weight)
                })

                # 保存交易评分权重
                conn.execute(text(query), {
                    'key': 'transaction_score',
                    'value': str(trans_weight)
                })

                # 保存履约评分权重
                conn.execute(text(query), {
                    'key': 'contract_score',
                    'value': str(contract_weight)
                })

        except Exception as e:
            logger.error(f"保存评分权重失败: {str(e)}")
            raise


def main():
    """主程序入口"""
    try:
        root = tk.Tk()
        app = CreditScoreApp(root)

        # 设置窗口图标（如果有的话）
        try:
            root.iconbitmap('icon.ico')
        except:
            pass

        # 运行应用
        root.mainloop()

    except Exception as e:
        logger.error(f"应用程序运行错误: {str(e)}")
        messagebox.showerror("错误", f"应用程序运行错误: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()