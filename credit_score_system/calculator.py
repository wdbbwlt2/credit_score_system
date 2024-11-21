# calculator.py - Part 1

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
    """评分计算器

    总分构成：
    - 基础评分：30分
    - 交易评分：30分
    - 履约评分：25分
    - 调整系数：0.9-1.5
    - 最终得分 = min((基础评分 + 交易评分 + 履约评分) * 调整系数, 100)
    """

    def __init__(self, db_manager: DatabaseManager):
        """初始化评分计算器"""
        self.db_manager = db_manager
        self.config_loader = ConfigLoader(db_manager)
        self.score_logger = ScoreLogger(db_manager)
        self.validator = DataValidator(db_manager)
        self.logger = logging.getLogger(__name__)
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
        """计算基础评分 (30分)"""
        try:
            # 1. 合同金额评分 (20分)
            customer_contracts = contract_df[contract_df['CUSTOMER'] == customer_code]

            # 防空判断前先进行数据类型转换
            if isinstance(customer_code, (int, float)):
                customer_code = str(customer_code)

            # 确保客户编号格式统一
            customer_code = customer_code.strip()

            total_contract_amount = 0
            if not customer_contracts.empty and 'CONTRACT AMT' in customer_contracts.columns:
                # 确保合同金额为数值类型
                customer_contracts['CONTRACT AMT'] = pd.to_numeric(customer_contracts['CONTRACT AMT'], errors='coerce')
                total_contract_amount = customer_contracts['CONTRACT AMT'].sum()

            contract_score = 0
            if total_contract_amount >= 500000:  # 50万以上
                contract_score = 20  # 从25分改为20分
            elif total_contract_amount >= 200000:  # 20万以上
                contract_score = 16  # 从20分改为16分
            elif total_contract_amount >= 100000:  # 10万以上
                contract_score = 12  # 从15分改为12分
            elif total_contract_amount >= 50000:  # 5万以上
                contract_score = 9  # 从12分改为9分
            elif total_contract_amount >= 10000:  # 1万以上
                contract_score = 6  # 从8分改为6分
            else:
                contract_score = 4  # 从5分改为4分

            # 2. 客户类型评分 (10分)
            code_column = 'CODE' if 'CODE' in collection_df.columns else 'code'
            customer_info = collection_df[collection_df[code_column] == customer_code]

            type_column = 'TYPE' if 'TYPE' in collection_df.columns else 'type'
            customer_type = customer_info[type_column].iloc[0] if not customer_info.empty else '其他'

            type_score = {
                '公立医院': 10,
                '连锁医院': 10,
                '私立机构': 8,
                '连锁经销商': 8,
                '公立经销商': 8,
                '法人客户': 8,
                '其他经销商': 6,
                '其他': 6,
                '天津法人': 8,
                '深圳法人': 8,
                '服务处': 7,
                '关联公司': 7,
                '集采配送公司': 8,
                '私立经销商': 7
            }.get(customer_type, 6)

            total_score = contract_score + type_score
            weighted_score = total_score * 0.30  # 从0.35改为0.30

            logger.debug(f"客户 {customer_code} 基础评分计算结果:")
            logger.debug(f"合同金额: {total_contract_amount}, 得分: {contract_score}")
            logger.debug(f"客户类型: {customer_type}, 得分: {type_score}")
            logger.debug(f"总分: {total_score}, 加权分: {weighted_score}")

            return {
                'score': total_score,
                'weighted_score': weighted_score,
                'weight': 0.30,  # 从0.35改为0.30
                'details': {
                    'contract_amount': float(total_contract_amount),
                    'contract_score': contract_score,
                    'customer_type': customer_type,
                    'type_score': type_score
                }
            }

        except Exception as e:
            logger.error(f"计算基础评分时发生错误: {str(e)}")
            # 返回默认最低分
            return {
                'score': 10,  # 最低基础分：合同4分 + 类型6分
                'weighted_score': 3.0,  # 10 * 0.30
                'weight': 0.30,
                'details': {
                    'contract_amount': 0,
                    'contract_score': 4,
                    'customer_type': '其他',
                    'type_score': 6
                }
            }

# calculator.py - Part 2

    def calculate_transaction_score(self, customer_code: str, sales_df: pd.DataFrame, year: int, month: int) -> Dict:
        """计算交易评分 (30分)"""
        try:
            # 数据预处理
            if isinstance(customer_code, (int, float)):
                customer_code = str(customer_code)
            customer_code = customer_code.strip()

            # 数据验证和预处理
            if sales_df is None or sales_df.empty:
                return self._get_default_transaction_score("无交易记录")

            # 确保必要的列存在
            required_columns = ['CUSTOMER', 'SALES AMT', 'SALES DATE']
            if not all(col in sales_df.columns for col in required_columns):
                return self._get_default_transaction_score("数据格式不完整")

            # 转换销售金额为数值类型
            sales_df['SALES AMT'] = pd.to_numeric(sales_df['SALES AMT'], errors='coerce')
            sales_df['SALES AMT'] = sales_df['SALES AMT'].fillna(0)

            # 确保日期格式正确
            sales_df['SALES DATE'] = pd.to_datetime(sales_df['SALES DATE'], errors='coerce')

            # 过滤指定年月的数据
            mask = (sales_df['SALES DATE'].dt.year == year) & (sales_df['SALES DATE'].dt.month == month)
            period_sales = sales_df[mask]

            customer_sales = period_sales[period_sales['CUSTOMER'] == customer_code].copy()
            if customer_sales.empty:
                return self._get_default_transaction_score("当期无交易记录")

            # 1. 月度交易金额评分 (17分)
            monthly_amount = customer_sales['SALES AMT'].sum()
            amount_score = 0
            if monthly_amount >= 200000:  # 20万以上
                amount_score = 17  # 从20分改为17分
            elif monthly_amount >= 100000:  # 10万以上
                amount_score = 14  # 从16分改为14分
            elif monthly_amount >= 50000:  # 5万以上
                amount_score = 10  # 从12分改为10分
            elif monthly_amount >= 10000:  # 1万以上
                amount_score = 7   # 从8分改为7分
            else:
                amount_score = 4   # 从5分改为4分

            # 2. 交易频率评分 (13分)
            transaction_days = len(customer_sales['SALES DATE'].dt.date.unique())
            frequency_score = 0
            if transaction_days >= 10:  # 10天以上
                frequency_score = 13  # 从15分改为13分
            elif transaction_days >= 6:  # 6天以上
                frequency_score = 10  # 从12分改为10分
            elif transaction_days >= 3:  # 3天以上
                frequency_score = 8   # 从9分改为8分
            elif transaction_days >= 1:  # 至少1天
                frequency_score = 5   # 从6分改为5分
            else:
                frequency_score = 3   # 维持3分不变

            total_score = amount_score + frequency_score
            weighted_score = total_score * 0.30  # 从0.35改为0.30

            logger.debug(f"客户 {customer_code} {year}年{month}月交易评分计算结果:")
            logger.debug(f"月度交易额: {monthly_amount}, 得分: {amount_score}")
            logger.debug(f"交易天数: {transaction_days}, 得分: {frequency_score}")
            logger.debug(f"总分: {total_score}, 加权分: {weighted_score}")

            return {
                'score': total_score,
                'weighted_score': weighted_score,
                'weight': 0.30,  # 从0.35改为0.30
                'details': {
                    'monthly_amount': float(monthly_amount),
                    'amount_score': amount_score,
                    'transaction_days': transaction_days,
                    'frequency_score': frequency_score
                }
            }

        except Exception as e:
            logger.error(f"计算交易评分时发生错误: {str(e)}")
            return self._get_default_transaction_score(f"计算错误: {str(e)}")

    def _get_default_transaction_score(self, reason: str) -> Dict:
        """获取默认交易评分"""
        return {
            'score': 7,  # 最低基础分：金额4分 + 频率3分
            'weighted_score': 2.1,  # 7 * 0.30 从2.8改为2.1
            'weight': 0.30,  # 从0.35改为0.30
            'details': {
                'reason': reason,
                'monthly_amount': 0,
                'amount_score': 4,
                'transaction_days': 0,
                'frequency_score': 3
            }
        }

# calculator.py - Part 3

    def calculate_contract_score(self, customer_code: str, contract_df: pd.DataFrame) -> Dict:
        """计算履约评分 (25分)"""
        try:
            # 数据预处理
            if isinstance(customer_code, (int, float)):
                customer_code = str(customer_code)
            customer_code = customer_code.strip()

            if contract_df is None or contract_df.empty:
                return self._get_default_contract_score("无合同记录")

            # 确保必要的列存在
            required_columns = ['CUSTOMER', 'CONTRACT AMT', 'COLLECTION', 'STATUS']
            if not all(col in contract_df.columns for col in required_columns):
                return self._get_default_contract_score("数据格式不完整")

            # 数据类型转换
            contract_df['CONTRACT AMT'] = pd.to_numeric(contract_df['CONTRACT AMT'], errors='coerce')
            contract_df['COLLECTION'] = pd.to_numeric(contract_df['COLLECTION'], errors='coerce')

            # 填充空值
            contract_df['CONTRACT AMT'] = contract_df['CONTRACT AMT'].fillna(0)
            contract_df['COLLECTION'] = contract_df['COLLECTION'].fillna(0)

            customer_contracts = contract_df[contract_df['CUSTOMER'] == customer_code]
            if customer_contracts.empty:
                return self._get_default_contract_score("无合同记录")

            # 1. 回款评分 (15分)
            total_amount = customer_contracts['CONTRACT AMT'].sum()
            collected_amount = customer_contracts['COLLECTION'].sum()

            # 防止除零错误
            collection_rate = (collected_amount / total_amount) if total_amount > 0 else 0

            payment_score = 0
            if collection_rate >= 0.9:  # 90%以上
                payment_score = 15  # 从20分改为15分
            elif collection_rate >= 0.8:  # 80%以上
                payment_score = 12  # 从16分改为12分
            elif collection_rate >= 0.7:  # 70%以上
                payment_score = 9   # 从12分改为9分
            elif collection_rate >= 0.6:  # 60%以上
                payment_score = 6   # 从8分改为6分
            else:
                payment_score = 4   # 从5分改为4分

            # 2. 合同完成率评分 (10分)
            # 统一状态大小写和去除空格
            customer_contracts['STATUS'] = customer_contracts['STATUS'].str.strip().str.upper()
            completed_statuses = ['CLOSE', 'COMPLETED', 'DONE', 'FINISHED']
            completed_contracts = len(customer_contracts[customer_contracts['STATUS'].isin(completed_statuses)])
            total_contracts = len(customer_contracts)

            # 防止除零错误
            completion_rate = completed_contracts / total_contracts if total_contracts > 0 else 0

            completion_score = 0
            if completion_rate >= 0.8:  # 80%以上
                completion_score = 10  # 维持10分不变
            elif completion_rate >= 0.7:  # 70%以上
                completion_score = 8   # 维持8分不变
            elif completion_rate >= 0.6:  # 60%以上
                completion_score = 6   # 维持6分不变
            elif completion_rate >= 0.5:  # 50%以上
                completion_score = 4   # 维持4分不变
            else:
                completion_score = 2   # 维持2分不变

            total_score = payment_score + completion_score
            weighted_score = total_score * 0.25  # 从0.30改为0.25

            logger.debug(f"客户 {customer_code} 履约评分计算结果:")
            logger.debug(f"回款率: {collection_rate:.2%}, 得分: {payment_score}")
            logger.debug(f"完成率: {completion_rate:.2%}, 得分: {completion_score}")
            logger.debug(f"总分: {total_score}, 加权分: {weighted_score}")

            return {
                'score': total_score,
                'weighted_score': weighted_score,
                'weight': 0.25,  # 从0.30改为0.25
                'details': {
                    'collection_rate': float(collection_rate),
                    'payment_score': payment_score,
                    'completion_rate': float(completion_rate),
                    'completion_score': completion_score,
                    'total_amount': float(total_amount),
                    'collected_amount': float(collected_amount),
                    'total_contracts': total_contracts,
                    'completed_contracts': completed_contracts
                }
            }

        except Exception as e:
            logger.error(f"计算履约评分时发生错误: {str(e)}")
            return self._get_default_contract_score(f"计算错误: {str(e)}")

    def _get_default_contract_score(self, reason: str) -> Dict:
        """获取默认履约评分"""
        return {
            'score': 6,  # 最低基础分：回款4分 + 完成率2分
            'weighted_score': 1.5,  # 6 * 0.25 从2.1改为1.5
            'weight': 0.25,  # 从0.30改为0.25
            'details': {
                'reason': reason,
                'collection_rate': 0,
                'payment_score': 4,
                'completion_rate': 0,
                'completion_score': 2,
                'total_amount': 0,
                'collected_amount': 0,
                'total_contracts': 0,
                'completed_contracts': 0
            }
        }

    def calculate_adjustment_factor(self, customer_code: str, sales_df: pd.DataFrame,
                                    contract_df: pd.DataFrame, year: int, month: int) -> Dict:
        """计算调整系数 (范围: 0.9-1.5)"""
        try:
            base_factor = 1.0
            adjustments = []

            # 数据预处理
            if isinstance(customer_code, (int, float)):
                customer_code = str(customer_code)
            customer_code = customer_code.strip()

            # 1. 合作年限奖励 (最高+0.2)
            cooperation_years = self._calculate_cooperation_years(customer_code, contract_df)
            if cooperation_years >= 5:  # 5年以上
                adjustments.append(('战略合作伙伴', 0.2))
            elif cooperation_years >= 3:  # 3年以上
                adjustments.append(('长期战略合作', 0.15))
            elif cooperation_years >= 2:  # 2年以上
                adjustments.append(('稳定合作', 0.1))
            elif cooperation_years >= 1:  # 1年以上
                adjustments.append(('常规合作', 0.05))

            # 2. 业务增长奖励 (最高+0.2)
            growth_rate = self._calculate_growth_rate(customer_code, sales_df, year, month)
            if growth_rate >= 0.5:  # 50%以上增长
                adjustments.append(('高速增长', 0.2))
            elif growth_rate >= 0.3:  # 30%以上增长
                adjustments.append(('快速增长', 0.15))
            elif growth_rate >= 0.1:  # 10%以上增长
                adjustments.append(('稳定增长', 0.1))
            elif growth_rate >= 0:  # 有增长
                adjustments.append(('小幅增长', 0.05))

            # 3. 交易规模奖励 (最高+0.1)
            if sales_df is not None and not sales_df.empty:
                sales_df['SALES AMT'] = pd.to_numeric(sales_df['SALES AMT'], errors='coerce')
                sales_mask = (pd.to_datetime(sales_df['SALES DATE']).dt.year == year) & \
                             (pd.to_datetime(sales_df['SALES DATE']).dt.month == month)
                monthly_amount = sales_df[sales_mask & (sales_df['CUSTOMER'] == customer_code)]['SALES AMT'].sum()

                if monthly_amount >= 1000000:  # 100万以上
                    adjustments.append(('超大额交易', 0.1))
                elif monthly_amount >= 500000:  # 50万以上
                    adjustments.append(('大额交易', 0.08))
                elif monthly_amount >= 200000:  # 20万以上
                    adjustments.append(('重要交易', 0.05))
                elif monthly_amount >= 100000:  # 10万以上
                    adjustments.append(('标准交易', 0.03))

            # 4. 计算最终系数
            total_adjustment = sum(factor for _, factor in adjustments)

            # 5. 限制最大调整幅度
            total_adjustment = min(total_adjustment, 0.5)  # 最多上调50%

            # 6. 计算最终系数
            final_factor = base_factor + total_adjustment

            # 7. 确保系数在合理范围内
            final_factor = max(0.9, min(1.5, final_factor))

            logger.debug(f"客户 {customer_code} 调整系数计算结果:")
            logger.debug(f"基础系数: {base_factor}")
            logger.debug(f"调整项: {adjustments}")
            logger.debug(f"总调整幅度: {total_adjustment}")
            logger.debug(f"最终系数: {final_factor}")

            return {
                'base_factor': base_factor,
                'final_factor': final_factor,
                'adjustments': adjustments,
                'details': {
                    'cooperation_years': float(cooperation_years),
                    'growth_rate': float(growth_rate) if not math.isnan(growth_rate) else 0.0,
                    'monthly_amount': float(monthly_amount) if 'monthly_amount' in locals() else 0.0,
                    'total_adjustment': float(total_adjustment),
                    'adjustment_breakdown': {
                        'years_bonus': next((factor for name, factor in adjustments if '合作' in name), 0.0),
                        'growth_bonus': next((factor for name, factor in adjustments if '增长' in name), 0.0),
                        'amount_bonus': next((factor for name, factor in adjustments if '交易' in name), 0.0)
                    }
                }
            }

        except Exception as e:
            logger.error(f"计算调整系数时发生错误: {str(e)}")
            # 返回默认调整系数
            return {
                'base_factor': 1.0,
                'final_factor': 1.0,
                'adjustments': [],
                'details': {
                    'cooperation_years': 0,
                    'growth_rate': 0.0,
                    'monthly_amount': 0.0,
                    'error': str(e),
                    'total_adjustment': 0.0,
                    'adjustment_breakdown': {
                        'years_bonus': 0.0,
                        'growth_bonus': 0.0,
                        'amount_bonus': 0.0
                    }
                }
            }


    # calculator.py - Part 4

    def calculate_final_score(self, base_score: Dict, transaction_score: Dict,
                              contract_score: Dict, adjustment_factor: Dict) -> Dict:
        """计算最终评分

        评分构成：
        - 基础评分：30分 (base_score)
        - 交易评分：30分 (transaction_score)
        - 履约评分：25分 (contract_score)
        - 调整系数：0.9-1.5 (adjustment_factor)
        - 最终得分 = min((基础评分 + 交易评分 + 履约评分) * 调整系数, 100)
        """
        try:
            # 确保所有输入数据有效
            if not all([base_score, transaction_score, contract_score, adjustment_factor]):
                raise ValueError("评分组成部分数据不完整")

            # 1. 获取各部分原始分数，确保数值有效
            base_raw = float(base_score.get('score', 0))
            transaction_raw = float(transaction_score.get('score', 0))
            contract_raw = float(contract_score.get('score', 0))

            # 2. 计算原始总分
            raw_score = base_raw + transaction_raw + contract_raw

            # 3. 获取调整系数
            factor = float(adjustment_factor.get('final_factor', 1.0))
            factor = max(0.9, min(1.5, factor))

            # 4. 计算调整后分数
            adjusted_score = raw_score * factor

            # 5. 限制最终分数在100分以内
            final_score = min(100, max(0, adjusted_score))
            final_score = self._round_decimal(final_score)

            # 6. 确定信用等级
            credit_level = self._determine_credit_level(final_score)

            # 记录详细的计算过程
            logger.debug(f"最终评分计算详情:")
            logger.debug(f"基础评分(30分制): {base_raw}")
            logger.debug(f"交易评分(30分制): {transaction_raw}")
            logger.debug(f"履约评分(25分制): {contract_raw}")
            logger.debug(f"原始总分(85分制): {raw_score}")
            logger.debug(f"调整系数: {factor}")
            logger.debug(f"调整后分数: {adjusted_score}")
            logger.debug(f"最终得分: {final_score}")
            logger.debug(f"信用等级: {credit_level}")

            return {
                'raw_score': float(raw_score),
                'final_score': float(final_score),
                'credit_level': credit_level,
                'calculation_details': {
                    'base_score': {
                        'raw': base_raw,
                        'percentage': f"{(base_raw / 30) * 100:.1f}%"
                    },
                    'transaction_score': {
                        'raw': transaction_raw,
                        'percentage': f"{(transaction_raw / 30) * 100:.1f}%"
                    },
                    'contract_score': {
                        'raw': contract_raw,
                        'percentage': f"{(contract_raw / 25) * 100:.1f}%"
                    },
                    'adjustment': {
                        'factor': factor,
                        'raw_score': raw_score,
                        'adjusted_score': adjusted_score,
                        'final_score': final_score,
                        'description': adjustment_factor.get('adjustments', [])
                    }
                }
            }

        except Exception as e:
            logger.error(f"计算最终评分时发生错误: {str(e)}")
            return {
                'raw_score': 0,
                'final_score': 0,
                'credit_level': 'C级',
                'calculation_details': {
                    'error': str(e)
                }
            }

    def verify_calculation(self, customer_code: str, year: int, month: int,
                           score_details: Dict) -> Dict:
        """验证评分计算结果"""
        try:
            validation_result = {
                'success': True,
                'errors': [],
                'warnings': []
            }

            # 基础数据验证
            if not score_details:
                validation_result['errors'].append("评分详情数据为空")
                validation_result['success'] = False
                return validation_result

            # 1. 检查评分范围
            try:
                final_score = float(score_details['final_score'])
                # 考虑到最低基础分的情况
                # 基础分10 + 交易分7 + 履约分6 = 23分
                min_score = 23 * 0.9  # 最低基础分乘以最低调整系数
                if not (min_score <= final_score <= 100):
                    validation_result['errors'].append(
                        f"最终得分 {final_score} 超出有效范围({min_score:.2f}-100)"
                    )
            except (ValueError, TypeError, KeyError):
                validation_result['errors'].append("最终得分数据无效")

            # 2. 检查评分组成
            try:
                base_score = float(score_details['base_score'])
                if not (10 <= base_score <= 30):  # 修改为新的分值范围
                    validation_result['errors'].append(
                        f"基础评分 {base_score} 超出有效范围(10-30)"
                    )
            except (ValueError, TypeError, KeyError):
                validation_result['errors'].append("基础评分数据无效")

            try:
                transaction_score = float(score_details['transaction_score'])
                if not (7 <= transaction_score <= 30):  # 修改为新的分值范围
                    validation_result['errors'].append(
                        f"交易评分 {transaction_score} 超出有效范围(7-30)"
                    )
            except (ValueError, TypeError, KeyError):
                validation_result['errors'].append("交易评分数据无效")

            try:
                contract_score = float(score_details['contract_score'])
                if not (6 <= contract_score <= 25):  # 修改为新的分值范围
                    validation_result['errors'].append(
                        f"履约评分 {contract_score} 超出有效范围(6-25)"
                    )
            except (ValueError, TypeError, KeyError):
                validation_result['errors'].append("履约评分数据无效")

            # 3. 检查调整系数
            try:
                adjustment_factor = float(score_details['adjustment_factor'])
                if not (0.9 <= adjustment_factor <= 1.5):
                    validation_result['errors'].append(
                        f"调整系数 {adjustment_factor} 超出有效范围(0.9-1.5)"
                    )
            except (ValueError, TypeError, KeyError):
                validation_result['errors'].append("调整系数数据无效")

            # 4. 检查信用等级
            valid_levels = ['AAA级', 'AA级', 'A级', 'BBB级', 'BB级', 'B级', 'C级']
            if score_details.get('credit_level') not in valid_levels:
                validation_result['errors'].append(
                    f"无效的信用等级: {score_details.get('credit_level')}"
                )

            # 5. 验证计算一致性
            if not validation_result['errors']:
                try:
                    expected_score = (base_score + transaction_score + contract_score) * adjustment_factor
                    expected_score = min(100, expected_score)  # 加入100分上限限制
                    if not math.isclose(expected_score, final_score, rel_tol=1e-2):
                        validation_result['errors'].append(
                            f"最终得分计算不一致: 期望值={expected_score:.2f}, 实际值={final_score:.2f}"
                        )
                except Exception as e:
                    validation_result['errors'].append(f"计算一致性验证失败: {str(e)}")

            # 6. 添加警告信息
            raw_total = base_score + transaction_score + contract_score
            if raw_total > 85:  # 三项总分超过85分
                validation_result['warnings'].append(
                    f"原始总分（{raw_total:.2f}）超过85分，请检查各项评分"
                )
            if final_score < 20:
                validation_result['warnings'].append(
                    f"评分异常偏低 ({final_score}), 请检查计算逻辑"
                )

            # 设置最终验证结果
            validation_result['success'] = len(validation_result['errors']) == 0

            return validation_result

        except Exception as e:
            logger.error(f"验证计算结果时发生错误: {str(e)}")
            return {
                'success': False,
                'errors': [str(e)],
                'warnings': []
            }
# calculator.py - Part 5 (Final)

    def process_and_verify(self, customer_code: str, year: int, month: int,
                           sales_df: pd.DataFrame, collection_df: pd.DataFrame,
                           contract_df: pd.DataFrame) -> Dict:
        """处理单个客户的评分计算并进行验证"""
        try:
            # 1. 计算基础评分 (30分)
            base_score = self.calculate_base_score(customer_code, contract_df, collection_df)

            # 2. 计算交易评分 (30分)
            transaction_score = self.calculate_transaction_score(customer_code, sales_df, year, month)

            # 3. 计算履约评分 (25分)
            contract_score = self.calculate_contract_score(customer_code, contract_df)

            # 4. 计算调整系数 (0.9-1.5)
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
                try:
                    score_id = self.save_score_result(customer_code, year, month, score_details)
                    if score_id:
                        logger.info(f"客户 {customer_code} 的评分结果已保存，ID: {score_id}")
                except Exception as e:
                    logger.error(f"保存评分结果失败: {str(e)}")
                    validation_result['warnings'].append("评分计算成功但保存失败")

            # 9. 记录评分结果的详细日志
            if validation_result['success']:
                logger.info(f"客户 {customer_code} {year}年{month}月评分计算成功:")
                logger.info(f"  基础评分(30分制): {base_score['score']:.2f}")
                logger.info(f"  交易评分(30分制): {transaction_score['score']:.2f}")
                logger.info(f"  履约评分(25分制): {contract_score['score']:.2f}")
                logger.info(f"  调整系数: {adjustment_factor['final_factor']:.2f}")
                logger.info(f"  最终得分: {final_score['final_score']:.2f}")
                logger.info(f"  信用等级: {final_score['credit_level']}")
            else:
                logger.warning(f"客户 {customer_code} {year}年{month}月评分计算未通过验证:")
                for error in validation_result['errors']:
                    logger.warning(f"  错误: {error}")

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
                if not math.isclose(old_score, new_score, rel_tol=1e-2):
                    self.score_logger.log_score_change(
                        customer_code=customer_code,
                        year=year,
                        month=month,
                        old_score=old_score,
                        new_score=new_score,
                        reason="月度评分更新"
                    )
                    logger.info(f"客户 {customer_code} 评分变化: {old_score:.2f} -> {new_score:.2f}")

            return score_id

        except Exception as e:
            logger.error(f"保存评分结果失败: {str(e)}")
            raise DatabaseError(f"保存评分结果失败: {str(e)}")

    def _determine_credit_level(self, score: float) -> str:
        """确定信用等级"""
        if not isinstance(score, (int, float)):
            return 'C级'

        try:
            score = float(score)
            if math.isnan(score):
                return 'C级'

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

        except Exception as e:
            logger.error(f"确定信用等级时发生错误: {str(e)}")
            return 'C级'


# Add these methods to the CreditScoreCalculator class in calculator.py

    def _calculate_cooperation_years(self, customer_code: str, contract_df: pd.DataFrame) -> float:
        """计算合作年限"""
        try:
            customer_contracts = contract_df[contract_df['CUSTOMER'] == customer_code]
            if customer_contracts.empty:
                return 0.0

            # 添加数据验证
            valid_dates = []
            for date_str in customer_contracts['CONTRACT DATE']:
                try:
                    # 过滤无效日期
                    if pd.notna(date_str) and date_str != '0000/00/00' and date_str != '':
                        date = pd.to_datetime(date_str)
                        if date.year > 1900:  # 确保日期在合理范围内
                            valid_dates.append(date)
                except Exception as e:
                    self.logger.warning(f"客户 {customer_code} 的合同日期 '{date_str}' 无效: {str(e)}")
                    continue

            if not valid_dates:
                return 0.0

            earliest_date = min(valid_dates)
            years = (datetime.now() - earliest_date).days / 365
            return round(years, 1)

        except Exception as e:
            self.logger.error(f"计算客户 {customer_code} 合作年限时发生错误: {str(e)}")
            return 0.0


    def _save_main_score(self, customer_code: str, year: int, month: int,
                         score_details: Dict) -> Optional[int]:
        """保存主表评分记录

        参数:
            customer_code: 客户编号
            year: 年份
            month: 月份
            score_details: 评分详情

        返回:
            Optional[int]: 评分记录ID
        """
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
                # 检查是否存在记录
                check_query = """
                    SELECT id FROM customer_credit_score
                    WHERE customer_code = :customer_code
                    AND score_year = :score_year
                    AND score_month = :score_month
                """
                existing = conn.execute(text(check_query), {
                    "customer_code": customer_code,
                    "score_year": year,
                    "score_month": month
                }).fetchone()

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

                self.logger.info(f"成功保存客户 {customer_code} 的 {year}年{month}月 评分记录")

        except Exception as e:
            self.logger.error(f"保存评分记录失败: {str(e)}")
            raise DatabaseError(f"保存评分记录失败: {str(e)}")


    def _get_customer_name(self, customer_code: str) -> str:
        """获取客户名称

        参数:
            customer_code: 客户编号

        返回:
            str: 客户名称
        """
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


    def _save_score_details(self, score_id: int, score_details: Dict, customer_code: str = None) -> None:
        """保存评分明细数据

        参数:
            score_id: 主表评分记录ID
            score_details: 评分详情数据字典
            customer_code: 客户编号(可选)
        """
        try:
            with self.db_manager.get_connection() as conn:
                # 1. 清除旧的明细记录
                delete_query = "DELETE FROM credit_score_detail WHERE score_id = :score_id"
                conn.execute(text(delete_query), {"score_id": score_id})

                # 2. 如果没有传入 customer_code，从数据库获取
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

                # 3. 准备插入明细数据的SQL
                insert_query = """
                    INSERT INTO credit_score_detail (
                        score_id,
                        customer_code,
                        dimension,
                        score_dimension,
                        score_item,
                        item_score,
                        weight,
                        weighted_score,
                        cal_formula,
                        data_snapshot,
                        details
                    ) VALUES (
                        :score_id,
                        :customer_code,
                        :dimension,
                        :score_dimension,
                        :score_item,
                        :item_score,
                        :weight,
                        :weighted_score,
                        :cal_formula,
                        :data_snapshot,
                        :details
                    )
                """

                # 4. 处理每个评分维度
                detail_records = []
                for dimension, details in score_details['score_details'].items():
                    # 跳过final_calculation，因为它已经包含在其他维度中
                    if dimension == 'final_calculation':
                        continue

                    dimension_name = {
                        'base_score': '基础评分',
                        'transaction_score': '交易评分',
                        'contract_score': '履约评分',
                        'adjustment_factor': '调整系数'
                    }.get(dimension, dimension)

                    if isinstance(details, dict):
                        record = {
                            "score_id": score_id,
                            "customer_code": customer_code,
                            "dimension": dimension,
                            "score_dimension": dimension_name,
                            "score_item": dimension,
                            "item_score": float(details.get('score', details.get('final_factor', 0))),
                            "weight": float(details.get('weight', 1.0)),
                            "weighted_score": float(details.get('weighted_score', 0)),
                            "cal_formula": f"{dimension}_calculation",
                            "data_snapshot": json.dumps(details.get('details', {}), ensure_ascii=False),
                            "details": json.dumps(details, ensure_ascii=False)
                        }
                        detail_records.append(record)

                # 5. 批量插入明细记录
                for record in detail_records:
                    conn.execute(text(insert_query), record)

                # 6. 提交事务
                conn.commit()
                self.logger.info(f"成功保存评分ID {score_id} 的明细记录")

        except Exception as e:
            self.logger.error(f"保存评分明细时发生错误: {str(e)}")
            raise DatabaseError(f"保存评分明细失败: {str(e)}")


    def _calculate_growth_rate(self, customer_code: str, sales_df: pd.DataFrame, year: int, month: int) -> float:
        """计算销售增长率

        参数:
            customer_code: 客户编号
            sales_df: 销售数据DataFrame
            year: 年份
            month: 月份

        返回:
            float: 销售增长率（小数形式）
        """
        try:
            if sales_df is None or sales_df.empty:
                return 0.0

            # 数据预处理
            sales_df['SALES DATE'] = pd.to_datetime(sales_df['SALES DATE'], errors='coerce')
            sales_df['SALES AMT'] = pd.to_numeric(sales_df['SALES AMT'], errors='coerce')
            sales_df['SALES AMT'] = sales_df['SALES AMT'].fillna(0)

            # 过滤客户数据
            customer_sales = sales_df[sales_df['CUSTOMER'] == customer_code]

            # 获取当月销售额
            current_month_mask = (customer_sales['SALES DATE'].dt.year == year) & \
                                 (customer_sales['SALES DATE'].dt.month == month)
            current_sales = customer_sales[current_month_mask]['SALES AMT'].sum()

            # 获取去年同月销售额
            last_year_mask = (customer_sales['SALES DATE'].dt.year == year - 1) & \
                             (customer_sales['SALES DATE'].dt.month == month)
            last_year_sales = customer_sales[last_year_mask]['SALES AMT'].sum()

            # 如果去年同期没有销售，返回0增长率
            if last_year_sales == 0:
                return 0.0

            # 计算增长率
            growth_rate = (current_sales - last_year_sales) / last_year_sales

            return growth_rate

        except Exception as e:
            self.logger.error(f"计算客户 {customer_code} 的销售增长率时发生错误: {str(e)}")
            return 0.0