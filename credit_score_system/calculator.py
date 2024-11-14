# calculator.py

import logging
import pandas as pd
import numpy as np
from datetime import datetime
import json
import math
from typing import Dict, List, Tuple, Optional
from sqlalchemy import text
from decimal import Decimal, ROUND_HALF_UP

from .validation import DataValidator
from .utils import (
    DatabaseManager,
    ConfigLoader,
    ScoreLogger,
    DataValidationError,
    ScoreCalculationError,
    DatabaseError,
    ConfigurationError
)

# 配置日志
logger = logging.getLogger(__name__)

class CreditScoreCalculator:
    def __init__(self, db_manager: DatabaseManager):
        """初始化评分计算器"""
        self.db_manager = db_manager
        self.config_loader = ConfigLoader(db_manager)
        self.score_logger = ScoreLogger(db_manager)
        self.validator = DataValidator(db_manager)
        # 添加 logger 初始化
        self.logger = logging.getLogger(__name__)  # 添加这行
        self.load_configs()
        logger.info("评分计算器初始化成功")

    def load_configs(self):
        """加载评分配置"""
        try:
            self.contract_amount_config = self.config_loader.get_config('contract_amount')
            self.customer_type_config = self.config_loader.get_config('customer_type')
            self.weight_config = self.config_loader.get_config('weight')
            self.credit_level_config = self.config_loader.get_config('credit_level')

            if not all([
                self.contract_amount_config,
                self.customer_type_config,
                self.weight_config,
                self.credit_level_config
            ]):
                raise ConfigurationError("一个或多个必需的配置项为空")

            logger.info("评分配置加载成功")
        except Exception as e:
            logger.error(f"加载评分配置失败: {str(e)}")
            raise ConfigurationError(f"评分配置加载失败: {str(e)}")

    def _round_decimal(self, value: float, places: int = 2) -> float:
        """使用Decimal进行精确四舍五入"""
        if isinstance(value, str):
            dec = Decimal(value)
        else:
            dec = Decimal(str(float(value)))
        return float(dec.quantize(Decimal(f'0.{"0" * places}'), rounding=ROUND_HALF_UP))

    def calculate_base_score(self, customer_code: str, contract_df: pd.DataFrame, collection_df: pd.DataFrame) -> Dict:
        """计算基础评分 (35分)"""
        try:
            # 1. 合同金额评分 (25分)
            customer_contracts = contract_df[contract_df['CUSTOMER'] == customer_code]

            # 增加防空判断
            if customer_contracts.empty:
                return {
                    'score': 0,
                    'weighted_score': 0,
                    'weight': 0.35,
                    'details': {
                        'contract_amount': 0,
                        'contract_score': 0,
                        'customer_type': '其他',
                        'type_score': 0
                    }
                }

            total_contract_amount = customer_contracts['CONTRACT AMT'].sum()

            contract_score = 0
            if total_contract_amount >= 500000:  # 50万以上
                contract_score = 25
            elif total_contract_amount >= 200000:  # 20万以上
                contract_score = 20
            elif total_contract_amount >= 100000:  # 10万以上
                contract_score = 15
            elif total_contract_amount >= 50000:  # 5万以上
                contract_score = 12
            elif total_contract_amount >= 10000:  # 1万以上
                contract_score = 8
            else:
                contract_score = 5

            # 2. 客户类型评分 (10分)
            customer_info = collection_df[collection_df['CODE'] == customer_code]
            customer_type = customer_info['TYPE'].iloc[0] if not customer_info.empty else '其他'

            type_score = {
                '公立医院': 10,
                '连锁医院': 10,
                '私立机构': 8,
                '连锁经销商': 8,
                '公立经销商': 8,
                '法人客户': 8,
                '其他经销商': 6,
                '其他': 6
            }.get(customer_type, 6)  # 提高基础分到6分

            total_score = contract_score + type_score
            weighted_score = total_score * 0.35  # 35%权重

            return {
                'score': total_score,
                'weighted_score': weighted_score,
                'weight': 0.35,
                'details': {
                    'contract_amount': total_contract_amount,
                    'contract_score': contract_score,
                    'customer_type': customer_type,
                    'type_score': type_score
                }
            }

        except Exception as e:
            logger.error(f"计算基础评分时发生错误: {str(e)}")
            return {
                'score': 0,
                'weighted_score': 0,
                'weight': 0.35,
                'details': {
                    'contract_amount': 0,
                    'contract_score': 0,
                    'customer_type': '其他',
                    'type_score': 0
                }
            }

    def calculate_transaction_score(self, customer_code: str, sales_df: pd.DataFrame, year: int, month: int) -> Dict:
        """计算交易评分 (35分)"""
        try:

            # 数据验证和预处理
            if sales_df is None or sales_df.empty:
                return {
                    'score': 0,
                    'weighted_score': 0,
                    'weight': 0.35,
                    'details': {'reason': '无交易记录'}
                }

            # 确保必要的列存在
            required_columns = ['CUSTOMER', 'SALES AMT', 'SALES DATE']
            if not all(col in sales_df.columns for col in required_columns):
                return {
                    'score': 0,
                    'weighted_score': 0,
                    'weight': 0.35,
                    'details': {'reason': '数据格式不完整'}
                }

            customer_sales = sales_df[sales_df['CUSTOMER'] == customer_code].copy()
            if customer_sales.empty:
                return {'score': 0, 'weighted_score': 0, 'weight': 0.35, 'details': {'reason': '无交易记录'}}

            # 1. 月度交易金额评分 (20分)
            monthly_amount = customer_sales['SALES AMT'].sum()
            amount_score = 0
            if monthly_amount >= 200000:  # 20万以上
                amount_score = 20
            elif monthly_amount >= 100000:  # 10万以上
                amount_score = 16
            elif monthly_amount >= 50000:  # 5万以上
                amount_score = 12
            elif monthly_amount >= 10000:  # 1万以上
                amount_score = 8
            else:
                amount_score = 5  # 提高基础分到5分

            # 2. 交易频率评分 (15分)
            transaction_days = len(customer_sales['SALES DATE'].unique())
            frequency_score = 0
            if transaction_days >= 10:  # 10天以上
                frequency_score = 15
            elif transaction_days >= 6:  # 6天以上
                frequency_score = 12
            elif transaction_days >= 3:  # 3天以上
                frequency_score = 9
            elif transaction_days >= 1:  # 至少1天
                frequency_score = 6
            else:
                frequency_score = 3

            total_score = amount_score + frequency_score
            weighted_score = total_score * 0.35  # 35%权重

            return {
                'score': total_score,
                'weighted_score': weighted_score,
                'weight': 0.35,
                'details': {
                    'monthly_amount': monthly_amount,
                    'amount_score': amount_score,
                    'transaction_days': transaction_days,
                    'frequency_score': frequency_score
                }
            }
        except Exception as e:
            logger.error(f"计算交易评分时发生错误: {str(e)}")
            return {
                'score': 0,
                'weighted_score': 0,
                'weight': 0.35,
                'details': {
                    'reason': f'计算错误: {str(e)}',
                    'monthly_amount': 0,
                    'amount_score': 0,
                    'transaction_days': 0,
                    'frequency_score': 0
                }
            }

    def calculate_contract_score(self, customer_code: str, contract_df: pd.DataFrame) -> Dict:
        """计算履约评分 (30分)"""
        try:

            if contract_df is None or contract_df.empty:
                return {
                    'score': 0,
                    'weighted_score': 0,
                    'weight': 0.3,
                    'details': {'reason': '无合同记录'}
                }

            # 确保必要的列存在
            required_columns = ['CUSTOMER', 'CONTRACT AMT', 'COLLECTION', 'STATUS']
            if not all(col in contract_df.columns for col in required_columns):
                return {
                    'score': 0,
                    'weighted_score': 0,
                    'weight': 0.3,
                    'details': {'reason': '数据格式不完整'}
                }

            customer_contracts = contract_df[contract_df['CUSTOMER'] == customer_code]
            if customer_contracts.empty:
                return {'score': 0, 'weighted_score': 0, 'weight': 0.3, 'details': {'reason': '无合同记录'}}

            # 1. 回款评分 (20分)
            total_amount = customer_contracts['CONTRACT AMT'].sum()
            collected_amount = customer_contracts['COLLECTION'].sum()
            collection_rate = collected_amount / total_amount if total_amount > 0 else 0

            payment_score = 0
            if collection_rate >= 0.9:  # 90%以上
                payment_score = 20
            elif collection_rate >= 0.8:  # 80%以上
                payment_score = 16
            elif collection_rate >= 0.7:  # 70%以上
                payment_score = 12
            elif collection_rate >= 0.6:  # 60%以上
                payment_score = 8
            else:
                payment_score = 5  # 提高基础分到5分

            # 2. 合同完成率评分 (10分)
            completed_contracts = len(customer_contracts[customer_contracts['STATUS'].isin(['CLOSE', 'COMPLETED'])])
            total_contracts = len(customer_contracts)
            completion_rate = completed_contracts / total_contracts if total_contracts > 0 else 0

            completion_score = 0
            if completion_rate >= 0.8:  # 80%以上
                completion_score = 10
            elif completion_rate >= 0.7:  # 70%以上
                completion_score = 8
            elif completion_rate >= 0.6:  # 60%以上
                completion_score = 6
            elif completion_rate >= 0.5:  # 50%以上
                completion_score = 4
            else:
                completion_score = 2

            total_score = payment_score + completion_score
            weighted_score = total_score * 0.3  # 30%权重

            return {
                'score': total_score,
                'weighted_score': weighted_score,
                'weight': 0.3,
                'details': {
                    'collection_rate': collection_rate,
                    'payment_score': payment_score,
                    'completion_rate': completion_rate,
                    'completion_score': completion_score
                }
            }

        except Exception as e:
            logger.error(f"计算履约评分时发生错误: {str(e)}")
            return {
                'score': 0,
                'weighted_score': 0,
                'weight': 0.3,
                'details': {
                    'reason': f'计算错误: {str(e)}',
                    'collection_rate': 0,
                    'payment_score': 0,
                    'completion_rate': 0,
                    'completion_score': 0
                }
            }

    def calculate_adjustment_factor(self, customer_code: str, sales_df: pd.DataFrame,
                                    contract_df: pd.DataFrame, year: int, month: int) -> Dict:
        """计算调整系数 (范围: 0.9-1.5)"""
        try:
            base_factor = 1.0
            adjustments = []

            # 1. 合作年限奖励 (最高+0.2)
            cooperation_years = self._calculate_cooperation_years(customer_code, contract_df)
            if cooperation_years >= 3:
                adjustments.append(('长期战略合作', 0.2))
            elif cooperation_years >= 2:
                adjustments.append(('稳定合作', 0.15))
            elif cooperation_years >= 1:
                adjustments.append(('常规合作', 0.1))

            # 2. 业务增长奖励 (最高+0.2)
            growth_rate = self._calculate_growth_rate(customer_code, sales_df, year, month)
            if growth_rate >= 0.3:  # 30%以上增长
                adjustments.append(('高速增长', 0.2))
            elif growth_rate >= 0.2:  # 20%以上增长
                adjustments.append(('快速增长', 0.15))
            elif growth_rate >= 0.1:  # 10%以上增长
                adjustments.append(('稳定增长', 0.1))

            # 3. 交易规模奖励 (最高+0.1)
            monthly_amount = sales_df[sales_df['CUSTOMER'] == customer_code]['SALES AMT'].sum()
            if monthly_amount >= 500000:  # 50万以上
                adjustments.append(('大额交易', 0.1))
            elif monthly_amount >= 200000:  # 20万以上
                adjustments.append(('重要交易', 0.08))
            elif monthly_amount >= 100000:  # 10万以上
                adjustments.append(('标准交易', 0.05))

            # 计算最终系数
            final_factor = base_factor + sum(factor for _, factor in adjustments)
            final_factor = max(0.9, min(1.5, final_factor))  # 限制在0.9-1.5之间

            return {
                'base_factor': base_factor,
                'final_factor': final_factor,
                'adjustments': adjustments,
                'details': {
                    'cooperation_years': cooperation_years,
                    'growth_rate': growth_rate,
                    'monthly_amount': monthly_amount
                }
            }

        except Exception as e:
            logger.error(f"计算调整系数时发生错误: {str(e)}")
            raise ScoreCalculationError(f"调整系数计算失败: {str(e)}")

    def _calculate_cooperation_years(self, customer_code: str, contract_df: pd.DataFrame) -> float:
        """计算合作年限"""
        try:
            customer_contracts = contract_df[contract_df['CUSTOMER'] == customer_code]
            if customer_contracts.empty:
                return 0.0

            earliest_date = pd.to_datetime(customer_contracts['CONTRACT DATE']).min()
            years = (datetime.now() - earliest_date).days / 365
            return round(years, 1)

        except Exception:
            return 0.0

    def _calculate_growth_rate(self, customer_code: str, sales_df: pd.DataFrame, year: int, month: int) -> float:
        """计算销售增长率"""
        try:
            customer_sales = sales_df[sales_df['CUSTOMER'] == customer_code]
            customer_sales['SALES DATE'] = pd.to_datetime(customer_sales['SALES DATE'])

            current_month_mask = (
                    (customer_sales['SALES DATE'].dt.year == year) &
                    (customer_sales['SALES DATE'].dt.month == month)
            )
            current_sales = customer_sales[current_month_mask]['SALES AMT'].sum()

            last_year_mask = (
                    (customer_sales['SALES DATE'].dt.year == year - 1) &
                    (customer_sales['SALES DATE'].dt.month == month)
            )
            last_year_sales = customer_sales[last_year_mask]['SALES AMT'].sum()

            if last_year_sales > 0:
                return (current_sales - last_year_sales) / last_year_sales
            return 0.0

        except Exception:
            return 0.0

    def _determine_credit_level(self, score: float) -> str:
        """确定信用等级"""
        if score >= 90:
            return 'AAA级'
        elif score >= 80:
            return 'AA级'
        elif score >= 70:
            return 'A级'
        elif score >= 60:
            return 'BBB级'
        elif score >= 50:
            return 'BB级'
        elif score >= 40:
            return 'B级'
        else:
            return 'C级'

    def verify_calculation(self, customer_code: str, year: int, month: int,
                           score_details: Dict) -> Dict:
        """验证评分计算结果"""
        try:
            validation_result = {
                'success': True,
                'errors': [],
                'warnings': []
            }

            # 1. 检查评分范围
            final_score = float(score_details['final_score'])
            if not (0 <= final_score <= 100):
                validation_result['errors'].append(
                    f"最终得分 {final_score} 超出有效范围(0-100)"
                )

            # 2. 检查评分组成
            base_score = float(score_details['base_score'])
            if not (0 <= base_score <= 35):  # 基础评分权重35%
                validation_result['errors'].append(
                    f"基础评分 {base_score} 超出有效范围(0-35)"
                )

            transaction_score = float(score_details['transaction_score'])
            if not (0 <= transaction_score <= 35):  # 交易评分权重35%
                validation_result['errors'].append(
                    f"交易评分 {transaction_score} 超出有效范围(0-35)"
                )

            contract_score = float(score_details['contract_score'])
            if not (0 <= contract_score <= 30):  # 履约评分权重30%
                validation_result['errors'].append(
                    f"履约评分 {contract_score} 超出有效范围(0-30)"
                )

            # 3. 检查调整系数
            adjustment_factor = float(score_details['adjustment_factor'])
            if not (0.8 <= adjustment_factor <= 1.5):  # 更新了调整系数范围
                validation_result['errors'].append(
                    f"调整系数 {adjustment_factor} 超出有效范围(0.8-1.5)"
                )

            # 4. 检查信用等级
            valid_levels = ['AAA级', 'AA级', 'A级', 'BBB级', 'BB级', 'B级', 'C级']
            if score_details['credit_level'] not in valid_levels:
                validation_result['errors'].append(
                    f"无效的信用等级: {score_details['credit_level']}"
                )

            # 5. 验证权重计算
            expected_score = (base_score + transaction_score + contract_score) * adjustment_factor
            if not math.isclose(expected_score, final_score, rel_tol=1e-2):
                validation_result['errors'].append(
                    f"最终得分计算不一致: 期望值={expected_score:.2f}, 实际值={final_score:.2f}"
                )

            # 设置验证结果
            validation_result['success'] = len(validation_result['errors']) == 0

            # 6. 分数异常偏低警告
            if final_score < 20:
                validation_result['warnings'].append(
                    f"评分异常偏低 ({final_score}), 请检查计算逻辑"
                )

            return validation_result

        except Exception as e:
            logger.error(f"验证计算结果时发生错误: {str(e)}")
            return {
                'success': False,
                'errors': [str(e)],
                'warnings': []
            }

    def calculate_final_score(self, base_score: Dict, transaction_score: Dict,
                              contract_score: Dict, adjustment_factor: Dict) -> Dict:
        """计算最终评分"""
        try:
            # 1. 计算加权分数
            weighted_base = base_score['score'] * 0.35  # 基础评分35%
            weighted_transaction = transaction_score['score'] * 0.35  # 交易评分35%
            weighted_contract = contract_score['score'] * 0.30  # 履约评分30%

            # 2. 计算原始总分
            raw_score = weighted_base + weighted_transaction + weighted_contract

            # 3. 应用调整系数
            final_score = raw_score * adjustment_factor['final_factor']
            final_score = self._round_decimal(final_score)

            # 4. 确保分数在合理范围内
            final_score = max(0, min(100, final_score))

            logger.debug(f"分数计算详情:")
            logger.debug(f"  基础评分: {base_score['score']} * 0.35 = {weighted_base}")
            logger.debug(f"  交易评分: {transaction_score['score']} * 0.35 = {weighted_transaction}")
            logger.debug(f"  履约评分: {contract_score['score']} * 0.30 = {weighted_contract}")
            logger.debug(f"  原始总分: {raw_score}")
            logger.debug(f"  调整系数: {adjustment_factor['final_factor']}")
            logger.debug(f"  最终得分: {final_score}")

            return {
                'raw_score': raw_score,
                'final_score': final_score,
                'credit_level': self._determine_credit_level(final_score)
            }

        except Exception as e:
            logger.error(f"计算最终评分时发生错误: {str(e)}")
            raise ScoreCalculationError(f"最终评分计算失败: {str(e)}")

    def process_and_verify(self, customer_code: str, year: int, month: int,
                           sales_df: pd.DataFrame, collection_df: pd.DataFrame,
                           contract_df: pd.DataFrame) -> Dict:
        """处理单个客户的评分计算并进行验证"""
        try:
            # 1. 计算基础评分
            base_score = self.calculate_base_score(customer_code, contract_df, collection_df)

            # 2. 计算交易评分
            transaction_score = self.calculate_transaction_score(customer_code, sales_df, year, month)

            # 3. 计算履约评分
            contract_score = self.calculate_contract_score(customer_code, contract_df)

            # 4. 计算调整系数
            adjustment_factor = self.calculate_adjustment_factor(customer_code, sales_df, contract_df, year, month)

            # 5. 计算最终分数
            final_score = self.calculate_final_score(base_score, transaction_score, contract_score, adjustment_factor)

            # 6. 整合所有得分信息
            score_details = {
                'base_score': base_score['score'],
                'transaction_score': transaction_score['score'],
                'contract_score': contract_score['score'],
                'adjustment_factor': adjustment_factor['final_factor'],
                'final_score': final_score['final_score'],
                'credit_level': final_score['credit_level'],
                'score_details': {
                    'base_score': base_score,
                    'transaction_score': transaction_score,
                    'contract_score': contract_score,
                    'adjustment_factor': adjustment_factor,
                    'final_calculation': final_score
                }
            }

            # 7. 验证计算结果
            validation_result = self.verify_calculation(customer_code, year, month, score_details)

            # 8. 如果验证通过，保存评分结果
            if validation_result['success']:
                score_id = self.save_score_result(customer_code, year, month, score_details)
                if score_id:
                    logger.info(f"客户 {customer_code} 的评分结果已保存，ID: {score_id}")

            return {
                'score_details': score_details,
                'validation_result': validation_result
            }

        except Exception as e:
            logger.error(f"处理客户 {customer_code} 的评分时发生错误: {str(e)}")
            return {
                'score_details': None,
                'validation_result': {
                    'success': False,
                    'errors': [str(e)],
                    'warnings': []
                }
            }

    def save_score_result(self, customer_code: str, year: int, month: int,
                          score_details: Dict) -> Optional[int]:
        """保存评分结果"""
        try:
            # 获取旧的评分记录（如果存在）
            query = """
                SELECT id, final_score 
                FROM customer_credit_score
                WHERE customer_code = :customer_code
                    AND score_year = :year
                    AND score_month = :month
                ORDER BY created_at DESC
                LIMIT 1
            """

            old_score = None
            with self.db_manager.get_connection() as conn:
                result = conn.execute(text(query), {
                    "customer_code": customer_code,
                    "year": year,
                    "month": month
                }).fetchone()

                if result:
                    old_score = result[1]

            # 保存主表评分记录
            score_id = self._save_main_score(customer_code, year, month, score_details)

            # 保存评分明细记录
            if score_id:
                self._save_score_details(
                    score_id=score_id,
                    score_details=score_details,
                    customer_code=customer_code
                )

            # 记录评分变化
            if old_score is not None:
                new_score = float(score_details['final_score'])
                self.score_logger.log_score_change(
                    customer_code=customer_code,
                    year=year,
                    month=month,
                    old_score=old_score,
                    new_score=new_score,
                    reason="月度评分更新"
                )

            return score_id

        except Exception as e:
            logger.error(f"保存评分结果失败: {str(e)}")
            raise DatabaseError(f"保存评分结果失败: {str(e)}")

    def _get_customer_name(self, customer_code: str) -> str:
        """获取客户名称"""
        try:
            query = """
                SELECT CONCAT(
                    CODE,
                    ' - ',
                    CASE TYPE
                        WHEN '公立医院' THEN '公立医院'
                        WHEN '连锁医院' THEN '连锁医院'
                        WHEN '私立机构' THEN '私立诊所'
                        WHEN '连锁经销商' THEN '连锁经销商'
                        WHEN '公立经销商' THEN '公立经销商'
                        WHEN '私立经销商' THEN '私立经销商'
                        WHEN '集采配送公司' THEN '集采配送'
                        WHEN '天津法人' THEN '津法人'
                        WHEN '深圳法人' THEN '深法人'
                        WHEN '服务处' THEN '服务处'
                        WHEN '关联公司' THEN '关联'
                        ELSE '其他'
                    END
                ) as customer_name
                FROM osstem_customer_list 
                WHERE CODE = :code
            """
            with self.db_manager.get_connection() as conn:
                result = conn.execute(text(query), {"code": customer_code}).fetchone()
                return result[0] if result else f"{customer_code} - 未知"
        except Exception as e:
            self.logger.error(f"获取客户名称时发生错误: {str(e)}")
            return f"{customer_code} - 未知"

    def _save_main_score(self, customer_code: str, year: int, month: int,
                         score_details: Dict) -> Optional[int]:
        """保存主表评分记录"""
        try:
            # 获取客户名称
            customer_name = self._get_customer_name(customer_code)

            # 构造基础数据
            base_data = {
                "customer_code": customer_code,
                "customer_name": customer_name,
                "score_year": year,
                "score_month": month,
                "base_score": self._round_decimal(float(score_details['base_score'])),
                "transaction_score": self._round_decimal(float(score_details['transaction_score'])),
                "contract_score": self._round_decimal(float(score_details['contract_score'])),
                "adjustment_factor": self._round_decimal(float(score_details['adjustment_factor'])),
                "final_score": self._round_decimal(float(score_details['final_score'])),
                "credit_level": str(score_details['credit_level'])
            }

            with self.db_manager.get_connection() as conn:
                # 先检查是否存在
                check_query = """
                    SELECT id FROM customer_credit_score
                    WHERE customer_code = :customer_code
                    AND score_year = :score_year
                    AND score_month = :score_month
                """
                existing = conn.execute(
                    text(check_query),
                    {
                        "customer_code": customer_code,
                        "score_year": year,
                        "score_month": month
                    }
                ).fetchone()

                if existing:
                    # 更新现有记录
                    update_query = """
                        UPDATE customer_credit_score
                        SET customer_name = :customer_name,
                            base_score = :base_score,
                            transaction_score = :transaction_score,
                            contract_score = :contract_score,
                            adjustment_factor = :adjustment_factor,
                            final_score = :final_score,
                            credit_level = :credit_level
                        WHERE customer_code = :customer_code
                        AND score_year = :score_year
                        AND score_month = :score_month
                    """
                    conn.execute(text(update_query), base_data)
                    return existing[0]
                else:
                    # 插入新记录
                    insert_query = """
                        INSERT INTO customer_credit_score (
                            customer_code, customer_name, score_year, score_month,
                            base_score, transaction_score, contract_score,
                            adjustment_factor, final_score, credit_level
                        ) VALUES (
                            :customer_code, :customer_name, :score_year, :score_month,
                            :base_score, :transaction_score, :contract_score,
                            :adjustment_factor, :final_score, :credit_level
                        )
                    """
                    result = conn.execute(text(insert_query), base_data)
                    conn.commit()
                    return result.lastrowid

            logger.info(f"成功保存客户 {customer_code} 的 {year}年{month}月 评分记录")

        except Exception as e:
            logger.error(f"保存评分记录失败: {str(e)}")
            raise DatabaseError(f"保存评分记录失败: {str(e)}")

    def _save_score_details(self, score_id: int, score_details: Dict, customer_code: str = None) -> None:
        """保存评分明细"""
        try:
            with self.db_manager.get_connection() as conn:
                # 清除旧的明细记录
                delete_query = "DELETE FROM credit_score_detail WHERE score_id = :score_id"
                conn.execute(text(delete_query), {"score_id": score_id})

                # 准备新的明细记录
                insert_query = """
                    INSERT INTO credit_score_detail (
                        score_id,
                        customer_code,
                        score_dimension,
                        score_item,
                        item_score,
                        weight,
                        weighted_score,
                        data_snapshot
                    ) VALUES (
                        :score_id,
                        :customer_code,
                        :dimension,
                        :item,
                        :score,
                        :weight,
                        :weighted_score,
                        :snapshot
                    )
                """

                # 如果没有传入 customer_code，尝试从数据库获取
                if not customer_code:
                    get_customer_query = """
                        SELECT customer_code 
                        FROM customer_credit_score 
                        WHERE id = :score_id
                    """
                    result = conn.execute(text(get_customer_query), {"score_id": score_id}).fetchone()
                    if result:
                        customer_code = result[0]
                    else:
                        raise DatabaseError(f"无法获取评分ID {score_id} 对应的客户编号")

                # 遍历每个维度的得分详情
                for dimension, details in score_details['score_details'].items():
                    dimension_name = {
                        'base_score': '基础评分',
                        'transaction_score': '交易评分',
                        'contract_score': '履约评分'
                    }.get(dimension, dimension)

                    # 处理每个评分项
                    if isinstance(details, dict):
                        params = {
                            "score_id": score_id,
                            "customer_code": customer_code,
                            "dimension": dimension_name,
                            "item": dimension,
                            "score": float(details.get('score', 0)),
                            "weight": float(details.get('weight', 0)),
                            "weighted_score": float(details.get('weighted_score', 0)),
                            "snapshot": json.dumps(details, ensure_ascii=False)
                        }
                        conn.execute(text(insert_query), params)

                conn.commit()
                self.logger.info(f"成功保存评分ID {score_id} 的明细记录")

        except Exception as e:
            self.logger.error(f"保存评分明细时发生错误: {str(e)}")
            raise DatabaseError(f"保存评分明细失败: {str(e)}")