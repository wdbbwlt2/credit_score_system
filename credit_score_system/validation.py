# credit_score_system/validation.py

import logging
import math
from typing import Dict, Optional
from sqlalchemy import text
from .utils import DatabaseManager, DatabaseError

logger = logging.getLogger(__name__)

class DataValidator:
    """数据完整性验证器"""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)

    def validate_score_data(
            self,
            customer_code: str,
            year: int,
            month: int,
            expected_scores: Dict
    ) -> Dict:
        """验证客户评分数据完整性

        参数:
            customer_code: 客户编号
            year: 年份
            month: 月份
            expected_scores: 预期的评分数据

        返回:
            Dict: 包含验证结果和详细信息的字典
        """
        validation_result = {
            'success': True,
            'main_table_status': False,
            'detail_table_status': False,
            'score_consistency': False,
            'errors': [],
            'warnings': []
        }

        try:
            # 1. 验证主表数据
            main_record = self._validate_main_table(
                customer_code, year, month, expected_scores
            )
            validation_result['main_table_status'] = bool(main_record)

            if not main_record:
                validation_result['success'] = False
                validation_result['errors'].append("主表记录未找到")
                return validation_result

            # 2. 验证明细表数据
            details_status = self._validate_detail_table(
                main_record['id'], expected_scores
            )
            validation_result['detail_table_status'] = details_status

            if not details_status:
                validation_result['success'] = False
                validation_result['errors'].append("评分明细数据不完整")

            # 3. 验证数据一致性
            consistency_result = self._validate_score_consistency(
                main_record, expected_scores
            )
            validation_result['score_consistency'] = consistency_result['consistent']

            if not consistency_result['consistent']:
                validation_result['success'] = False
                validation_result['errors'].extend(consistency_result['discrepancies'])

            # 4. 检查数值合理性
            reasonability_result = self._validate_score_reasonability(main_record)
            if not reasonability_result['reasonable']:
                validation_result['warnings'].extend(reasonability_result['warnings'])

            return validation_result

        except Exception as e:
            self.logger.error(f"数据验证过程中发生错误: {str(e)}")
            validation_result['success'] = False
            validation_result['errors'].append(f"验证过程发生错误: {str(e)}")
            return validation_result

    def _validate_main_table(
            self,
            customer_code: str,
            year: int,
            month: int,
            expected_scores: Dict
    ) -> Optional[Dict]:
        """验证主表数据"""
        query = """
            SELECT id, customer_code, customer_name, 
                   base_score, transaction_score, contract_score,
                   adjustment_factor, final_score, credit_level
            FROM customer_credit_score
            WHERE customer_code = :customer_code
                AND score_year = :year
                AND score_month = :month
        """

        try:
            with self.db_manager.get_connection() as conn:
                result = conn.execute(text(query), {
                    "customer_code": customer_code,
                    "year": year,
                    "month": month
                }).fetchone()

                if result:
                    return dict(result._mapping)
                return None
        except Exception as e:
            self.logger.error(f"验证主表数据时发生错误: {str(e)}")
            return None
# validation.py - Part 2

def _validate_detail_table(self, score_id: int, expected_scores: Dict) -> bool:
    """验证明细表数据"""
    query = """
        SELECT COUNT(*) as detail_count
        FROM credit_score_detail
        WHERE score_id = :score_id
    """

    try:
        with self.db_manager.get_connection() as conn:
            result = conn.execute(text(query), {"score_id": score_id}).fetchone()
            expected_detail_count = len(expected_scores.get('score_details', {}))
            return result[0] == expected_detail_count
    except Exception as e:
        self.logger.error(f"验证明细表数据时发生错误: {str(e)}")
        return False

def _validate_score_consistency(
        self,
        stored_record: Dict,
        expected_scores: Dict
) -> Dict:
    """验证评分数据一致性"""
    discrepancies = []
    score_fields = [
        ('base_score', 'base_score'),
        ('transaction_score', 'transaction_score'),
        ('contract_score', 'contract_score'),
        ('adjustment_factor', 'adjustment_factor'),
        ('final_score', 'final_score'),
        ('credit_level', 'credit_level')
    ]

    for db_field, expected_field in score_fields:
        stored_value = float(stored_record[db_field]) if db_field != 'credit_level' else stored_record[db_field]
        expected_value = float(expected_scores[expected_field]) if db_field != 'credit_level' else expected_scores[
            expected_field]

        if db_field == 'credit_level':
            if stored_value != expected_value:
                discrepancies.append(f"{db_field}: 存储值={stored_value}, 预期值={expected_value}")
        else:
            if not math.isclose(stored_value, expected_value, rel_tol=1e-5):
                discrepancies.append(f"{db_field}: 存储值={stored_value}, 预期值={expected_value}")

    return {
        'consistent': len(discrepancies) == 0,
        'discrepancies': discrepancies
    }

def _validate_score_reasonability(self, record: Dict) -> Dict:
    """验证评分数值合理性"""
    warnings = []

    try:
        final_score = float(record['final_score'])

        # 放宽评分范围检查的限制
        if not (0 <= final_score <= 100):
            warnings.append(f"最终评分超出正常范围: {final_score}")

        # 增加各分项的合理性检查
        if float(record['base_score']) > 35:  # 基础评分最高35分
            warnings.append(f"基础评分超出范围: {record['base_score']}")

        if float(record['transaction_score']) > 35:  # 交易评分最高35分
            warnings.append(f"交易评分超出范围: {record['transaction_score']}")

        if float(record['contract_score']) > 30:  # 履约评分最高30分
            warnings.append(f"履约评分超出范围: {record['contract_score']}")
    except ValueError as e:
        warnings.append(f"评分数值转换错误: {str(e)}")

    return {
        'reasonable': len(warnings) == 0,
        'warnings': warnings
    }

def validate_batch_scores(
        self,
        year: int,
        month: int,
        sample_size: int = 100
) -> Dict:
    """验证批量评分结果

    参数:
        year: 年份
        month: 月份
        sample_size: 抽样验证的客户数量

    返回:
        Dict: 批量验证结果
    """
    query = """
        SELECT customer_code, final_score, credit_level
        FROM customer_credit_score
        WHERE score_year = :year
        AND score_month = :month
        ORDER BY RAND()
        LIMIT :sample_size
    """

    try:
        results = {
            'total_checked': 0,
            'passed': 0,
            'failed': 0,
            'error_samples': []
        }

        with self.db_manager.get_connection() as conn:
            sample_records = conn.execute(
                text(query),
                {"year": year, "month": month, "sample_size": sample_size}
            ).fetchall()

            for record in sample_records:
                customer_code = record['customer_code']
                validation_result = self.validate_score_data(
                    customer_code=customer_code,
                    year=year,
                    month=month,
                    expected_scores=self._recalculate_scores(
                        customer_code, year, month
                    )
                )

                results['total_checked'] += 1
                if validation_result['success']:
                    results['passed'] += 1
                else:
                    results['failed'] += 1
                    results['error_samples'].append({
                        'customer_code': customer_code,
                        'errors': validation_result['errors']
                    })

        return results

    except Exception as e:
        self.logger.error(f"批量验证评分时发生错误: {str(e)}")
        raise ValidationError(f"批量验证失败: {str(e)}")