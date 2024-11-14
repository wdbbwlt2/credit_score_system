# test.py - Part 1

import os
import sys
import json
from pathlib import Path
import logging
from typing import List, Dict, Optional, Any
import pandas as pd
import math
from decimal import Decimal, ROUND_HALF_UP

# 设置日志
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'extended_test.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

from credit_score_system import (
    CreditScoreSystem,
    DatabaseConfig,
    DatabaseManager,
    DataValidator,
    CreditScoreCalculator
)


class CreditScoreSystemTester:
    def __init__(self, config_path: str):
        """初始化测试器

        参数:
            config_path: 配置文件路径
        """
        try:
            self.load_config(config_path)
            self.initialize_system()
            logger.info("测试器初始化成功")
        except Exception as e:
            logger.error(f"测试器初始化失败: {str(e)}")
            raise

    def load_config(self, config_path: str) -> None:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        except Exception as e:
            logger.error(f"加载配置文件失败: {str(e)}")
            raise

    def initialize_system(self) -> None:
        """初始化评分系统"""
        try:
            self.system = CreditScoreSystem(self.config)
            self.calculator = self.system.calculator
        except Exception as e:
            logger.error(f"初始化评分系统失败: {str(e)}")
            raise
# test.py - Part 2

    def validate_test_results(self, result: Dict) -> bool:
        """验证测试结果的正确性"""
        try:
            # 如果结果为None或缺少必要字段，返回False
            if result is None or 'score_details' not in result or result['score_details'] is None:
                logger.error("测试结果为空或缺少score_details")
                return False

            score_details = result['score_details']

            # 1. 基本数据完整性检查
            required_fields = [
                'base_score', 'transaction_score', 'contract_score',
                'adjustment_factor', 'final_score', 'credit_level'
            ]
            for field in required_fields:
                if field not in score_details:
                    logger.error(f"缺少必要字段: {field}")
                    return False

            # 2. 评分范围检查
            final_score = float(score_details['final_score'])
            if not 0 <= final_score <= 100:
                logger.error(f"最终得分超出范围: {final_score}")
                return False

            # 3. 评分明细数据检查
            if 'score_details' not in score_details:
                logger.error("缺少评分明细数据")
                return False

            # 4. 调整系数范围检查
            adjustment_factor = float(score_details['adjustment_factor'])
            if not 0.8 <= adjustment_factor <= 1.2:
                logger.error(f"调整系数超出范围: {adjustment_factor}")
                return False

            # 5. 信用等级合法性检查
            valid_levels = ['AAA级', 'AA级', 'A级', 'BBB级', 'BB级', 'B级', 'C级']
            if score_details['credit_level'] not in valid_levels:
                logger.error(f"无效的信用等级: {score_details['credit_level']}")
                return False

            # 6. 验证结果检查
            if 'validation_result' not in result or not result['validation_result'].get('success', False):
                if 'validation_result' in result and 'errors' in result['validation_result']:
                    logger.error(f"验证失败: {result['validation_result']['errors']}")
                else:
                    logger.error("验证结果缺失或验证失败")
                return False

            logger.info("测试结果验证通过")
            return True

        except Exception as e:
            logger.error(f"验证测试结果时发生错误: {str(e)}")
            return False

    def test_single_customer(self, customer_code: str, year: int, month: int) -> Dict:
        """测试单个客户的评分计算"""
        try:
            logger.info(f"开始测试客户 {customer_code} 的评分计算")

            # 加载数据
            sales_df, collection_df, contract_df = self.system.data_loader.load_monthly_data(
                year, month
            )

            # 计算并验证评分
            result = self.calculator.process_and_verify(
                customer_code=customer_code,
                year=year,
                month=month,
                sales_df=sales_df,
                collection_df=collection_df,
                contract_df=contract_df
            )

            # 添加结果验证
            if not self.validate_test_results(result):
                logger.warning(f"客户 {customer_code} 的测试结果验证失败")
                result['validation_result']['success'] = False
                if 'errors' not in result['validation_result']:
                    result['validation_result']['errors'] = []
                result['validation_result']['errors'].append("测试结果验证失败")

            self._print_test_result(customer_code, result)
            return result

        except Exception as e:
            logger.error(f"测试客户 {customer_code} 时发生错误: {str(e)}")
            return {
                'score_details': None,
                'validation_result': {
                    'success': False,
                    'errors': [str(e)]
                }
            }

    def _print_test_result(self, customer_code: str, result: Dict) -> None:
        """打印测试结果"""
        logger.info("=" * 50)
        logger.info(f"客户 {customer_code} 测试结果:")

        if result is None or result.get('score_details') is None:
            logger.error(f"客户 {customer_code} 的评分计算失败")
            if result and result.get('validation_result'):
                logger.error(f"错误信息: {result['validation_result'].get('errors', ['未知错误'])}")
            logger.info("=" * 50)
            return

        try:
            score_details = result['score_details']
            logger.info(f"基础评分: {score_details['base_score']}")
            logger.info(f"交易评分: {score_details['transaction_score']}")
            logger.info(f"履约评分: {score_details['contract_score']}")
            logger.info(f"调整系数: {score_details['adjustment_factor']}")
            logger.info(f"最终评分: {score_details['final_score']}")
            logger.info(f"信用等级: {score_details['credit_level']}")
            logger.info(f"验证状态: {'通过' if result['validation_result']['success'] else '失败'}")

            if not result['validation_result']['success']:
                logger.warning(f"验证错误: {result['validation_result']['errors']}")
            if result['validation_result'].get('warnings'):
                logger.warning(f"警告信息: {result['validation_result']['warnings']}")

        except Exception as e:
            logger.error(f"打印测试结果时发生错误: {str(e)}")

        logger.info("=" * 50)
# test.py - Part 2

    def validate_test_results(self, result: Dict) -> bool:
        """验证测试结果的正确性"""
        try:
            # 如果结果为None或缺少必要字段，返回False
            if result is None or 'score_details' not in result or result['score_details'] is None:
                logger.error("测试结果为空或缺少score_details")
                return False

            score_details = result['score_details']

            # 1. 基本数据完整性检查
            required_fields = [
                'base_score', 'transaction_score', 'contract_score',
                'adjustment_factor', 'final_score', 'credit_level'
            ]
            for field in required_fields:
                if field not in score_details:
                    logger.error(f"缺少必要字段: {field}")
                    return False

            # 2. 评分范围检查
            final_score = float(score_details['final_score'])
            if not 0 <= final_score <= 100:
                logger.error(f"最终得分超出范围: {final_score}")
                return False

            # 3. 评分明细数据检查
            if 'score_details' not in score_details:
                logger.error("缺少评分明细数据")
                return False

            # 4. 调整系数范围检查
            adjustment_factor = float(score_details['adjustment_factor'])
            if not 0.8 <= adjustment_factor <= 1.2:
                logger.error(f"调整系数超出范围: {adjustment_factor}")
                return False

            # 5. 信用等级合法性检查
            valid_levels = ['AAA级', 'AA级', 'A级', 'BBB级', 'BB级', 'B级', 'C级']
            if score_details['credit_level'] not in valid_levels:
                logger.error(f"无效的信用等级: {score_details['credit_level']}")
                return False

            # 6. 验证结果检查
            if 'validation_result' not in result or not result['validation_result'].get('success', False):
                if 'validation_result' in result and 'errors' in result['validation_result']:
                    logger.error(f"验证失败: {result['validation_result']['errors']}")
                else:
                    logger.error("验证结果缺失或验证失败")
                return False

            logger.info("测试结果验证通过")
            return True

        except Exception as e:
            logger.error(f"验证测试结果时发生错误: {str(e)}")
            return False

    def test_single_customer(self, customer_code: str, year: int, month: int) -> Dict:
        """测试单个客户的评分计算"""
        try:
            logger.info(f"开始测试客户 {customer_code} 的评分计算")

            # 加载数据
            sales_df, collection_df, contract_df = self.system.data_loader.load_monthly_data(
                year, month
            )

            # 计算并验证评分
            result = self.calculator.process_and_verify(
                customer_code=customer_code,
                year=year,
                month=month,
                sales_df=sales_df,
                collection_df=collection_df,
                contract_df=contract_df
            )

            # 添加结果验证
            if not self.validate_test_results(result):
                logger.warning(f"客户 {customer_code} 的测试结果验证失败")
                result['validation_result']['success'] = False
                if 'errors' not in result['validation_result']:
                    result['validation_result']['errors'] = []
                result['validation_result']['errors'].append("测试结果验证失败")

            self._print_test_result(customer_code, result)
            return result

        except Exception as e:
            logger.error(f"测试客户 {customer_code} 时发生错误: {str(e)}")
            return {
                'score_details': None,
                'validation_result': {
                    'success': False,
                    'errors': [str(e)]
                }
            }

    def _print_test_result(self, customer_code: str, result: Dict) -> None:
        """打印测试结果"""
        logger.info("=" * 50)
        logger.info(f"客户 {customer_code} 测试结果:")

        if result is None or result.get('score_details') is None:
            logger.error(f"客户 {customer_code} 的评分计算失败")
            if result and result.get('validation_result'):
                logger.error(f"错误信息: {result['validation_result'].get('errors', ['未知错误'])}")
            logger.info("=" * 50)
            return

        try:
            score_details = result['score_details']
            logger.info(f"基础评分: {score_details['base_score']}")
            logger.info(f"交易评分: {score_details['transaction_score']}")
            logger.info(f"履约评分: {score_details['contract_score']}")
            logger.info(f"调整系数: {score_details['adjustment_factor']}")
            logger.info(f"最终评分: {score_details['final_score']}")
            logger.info(f"信用等级: {score_details['credit_level']}")
            logger.info(f"验证状态: {'通过' if result['validation_result']['success'] else '失败'}")

            if not result['validation_result']['success']:
                logger.warning(f"验证错误: {result['validation_result']['errors']}")
            if result['validation_result'].get('warnings'):
                logger.warning(f"警告信息: {result['validation_result']['warnings']}")

        except Exception as e:
            logger.error(f"打印测试结果时发生错误: {str(e)}")

        logger.info("=" * 50)