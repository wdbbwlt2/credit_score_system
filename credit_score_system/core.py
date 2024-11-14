# credit_score_system/core.py
from typing import Dict, List
from datetime import datetime
import logging
from pathlib import Path
import json

from .utils import DatabaseConfig, DatabaseManager, DataLoader, ConfigLoader, ScoreLogger
from .calculator import CreditScoreCalculator

logger = logging.getLogger(__name__)

class CreditScoreSystem:
    def __init__(self, config: Dict):
        self.config = config
        self.setup_components()

    def setup_components(self):
        try:
            db_config = DatabaseConfig(
                host=self.config['database']['host'],
                user=self.config['database']['user'],
                password=self.config['database']['password'],
                database=self.config['database']['database'],
                port=self.config['database'].get('port', 3306)
            )

            self.db_manager = DatabaseManager(db_config)
            self.data_loader = DataLoader(self.db_manager)
            self.config_loader = ConfigLoader(self.db_manager)
            self.score_logger = ScoreLogger(self.db_manager)
            self.calculator = CreditScoreCalculator(self.db_manager)

            logger.info("系统组件初始化完成")
        except Exception as e:
            logger.error(f"初始化系统组件失败: {str(e)}")
            raise

    def process_history(self, start_year: int, end_year: int) -> Dict[str, List[Dict]]:
        """处理历史数据评分"""
        logger.info(f"开始处理 {start_year}-{end_year} 年度的历史评分")
        print(f"开始处理 {start_year}-{end_year} 年度的历史评分")
        results = {}
        total_customers = 0
        processed_customers = 0

        try:
            for year in range(start_year, end_year + 1):
                results[year] = {}
                for month in range(1, 13):
                    if year == datetime.now().year and month > datetime.now().month:
                        break

                    logger.info(f"开始处理 {year}年{month}月 的数据")
                    print(f"处理 {year}年{month}月 的数据")

                    try:
                        month_results = self.process_month(year, month)
                        results[year][month] = month_results

                        # 统计处理结果
                        success_count = sum(1 for r in month_results if r.get('success', False))
                        fail_count = len(month_results) - success_count
                        processed_customers += len(month_results)

                        logger.info(f"{year}年{month}月处理完成，成功: {success_count}，失败: {fail_count}")
                        print(f"{year}年{month}月处理完成，成功: {success_count}，失败: {fail_count}")

                    except Exception as e:
                        logger.error(f"处理 {year}年{month}月 评分时发生错误: {str(e)}")
                        results[year][month] = []
                        continue

            logger.info(f"历史数据处理完成，共处理 {processed_customers} 个客户评分")
            return results

        except Exception as e:
            logger.error(f"处理历史数据时发生错误: {str(e)}")
            raise

    def process_month(self, year: int, month: int) -> List[Dict]:
        """处理单个月份的数据"""
        try:
            logger.info(f"开始处理 {year}年{month}月 的数据")

            # 加载数据
            sales_df, collection_df, contract_df = self.data_loader.load_monthly_data(year, month)

            # 获取活跃客户
            active_customers = self.data_loader.get_active_customers(year, month)
            logger.info(f"{year}年{month}月共有 {len(active_customers)} 个活跃客户")

            # 处理每个客户
            results = []
            success_count = 0
            fail_count = 0

            for customer_code in active_customers:
                try:
                    # 使用新的 process_and_verify 方法
                    result = self.calculator.process_and_verify(
                        customer_code=customer_code,
                        year=year,
                        month=month,
                        sales_df=sales_df,
                        collection_df=collection_df,
                        contract_df=contract_df
                    )

                    results.append({
                        'customer_code': customer_code,
                        'score': result['score_details'],
                        'validation': result['validation_result'],
                        'success': result['validation_result']['success']
                    })

                    if result['validation_result']['success']:
                        success_count += 1
                    else:
                        fail_count += 1

                    if success_count % 100 == 0:
                        logger.info(f"已成功处理 {success_count} 个客户")

                except Exception as e:
                    logger.error(f"处理客户 {customer_code} 时发生错误: {str(e)}")
                    results.append({
                        'customer_code': customer_code,
                        'error': str(e),
                        'success': False
                    })
                    fail_count += 1
                    continue

            logger.info(f"{year}年{month}月处理完成，成功: {success_count}，失败: {fail_count}")
            return results

        except Exception as e:
            logger.error(f"处理 {year}年{month}月 数据时发生错误: {str(e)}")
            raise