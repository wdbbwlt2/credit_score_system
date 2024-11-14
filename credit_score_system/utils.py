# utils.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Tuple, List, Dict, Optional, Any
import sqlalchemy
from sqlalchemy import create_engine, text
from contextlib import contextmanager
from dataclasses import dataclass
import json
import warnings
from pandas.tseries.offsets import MonthEnd
import os

warnings.filterwarnings('ignore')  # 忽略警告信息

# 创建日志目录
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'credit_score.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# 自定义异常类
class CreditScoreException(Exception):
    """信用评分系统基础异常类"""
    pass


class DataValidationError(CreditScoreException):
    """数据验证异常"""
    pass


class ScoreCalculationError(CreditScoreException):
    """评分计算异常"""
    pass


class DatabaseError(CreditScoreException):
    """数据库操作异常"""
    pass


class ConfigurationError(CreditScoreException):
    """配置加载异常"""
    pass


@dataclass
class DatabaseConfig:
    """数据库配置类

    属性:
        host: 数据库主机地址
        user: 数据库用户名
        password: 数据库密码
        database: 数据库名称
        port: 数据库端口号，默认3306
    """
    host: str
    user: str
    password: str
    database: str
    port: int = 3306

    def get_connection_string(self) -> str:
        """获取数据库连接字符串"""
        return f'mysql+pymysql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}'


class DatabaseManager:
    """数据库管理类"""

    def __init__(self, config: DatabaseConfig):
        """初始化数据库管理器

        参数:
            config: 数据库配置对象
        """
        self.config = config
        self._engine = None
        self.setup_connection_pool()

    def test_connection(self) -> bool:
        """测试数据库连接"""
        try:
            with self.get_connection() as conn:
                conn.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.error(f"数据库连接测试失败: {str(e)}")
            return False

    def setup_connection_pool(self):
        """设置数据库连接池"""
        try:
            self._engine = create_engine(
                self.config.get_connection_string(),
                pool_size=5,  # 连接池大小
                max_overflow=10,  # 最大溢出连接数
                pool_timeout=30,  # 连接超时时间
                pool_recycle=3600  # 连接回收时间
            )
            logger.info("数据库连接池初始化成功")
        except Exception as e:
            logger.error(f"数据库连接池初始化失败: {str(e)}")
            raise DatabaseError("数据库连接池初始化失败")

    @contextmanager
    def get_connection(self):
        """获取数据库连接（上下文管理器）"""
        if self._engine is None:
            raise DatabaseError("数据库引擎未初始化")

        conn = self._engine.connect()
        try:
            yield conn
        except Exception as e:
            logger.error(f"数据库连接错误: {str(e)}")
            raise DatabaseError(f"数据库操作失败: {str(e)}")
        finally:
            conn.close()


class DataLoader:
    """数据加载类"""

    def __init__(self, db_manager: DatabaseManager):
        """初始化数据加载器

        参数:
            db_manager: 数据库管理器实例
        """
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)

    def load_monthly_data(self, year: int, month: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """加载指定月份的数据

        参数:
            year: 年份
            month: 月份

        返回:
            tuple: (销售数据, 客户数据, 合同数据)的元组
        """
        try:
            # 验证日期有效性
            if not CreditScoreUtils.validate_date(year, month):
                raise DataValidationError(f"无效的日期: {year}-{month}")

            # 格式化年月字符串
            year_month = f"{year:04d}-{month:02d}"

            # 定义SQL查询
            queries = {
                'sales': f"""
                    SELECT 
                        `SALES DATE`,
                        `SALES NO`,
                        `ITEM CODE`,
                        `ITEM NAME`,
                        `TRANSACTION`,
                        `SALES QTY`,
                        `SALES PRICE`,
                        `SALES AMT`,
                        `CST CODE` as CUSTOMER,
                        `CONTRACT NO`,
                        `HIGH DEPT`,
                        `DEPARTMENT`,
                        `IN-CHARGE`
                    FROM osstem_sales_list 
                    WHERE DATE_FORMAT(`SALES DATE`, '%Y-%m') = '{year_month}'
                """,
                'collection': """
                    SELECT 
                        CODE,
                        `1` as province,
                        `2` as city,
                        `3` as service_dept,
                        TYPE
                    FROM osstem_customer_list
                """,
                'contract': f"""
                    SELECT 
                        `DATE`,
                        CUSTOMER,
                        CUSTOMER_NAME,
                        `CONTRACT No`,
                        `CONTRACT AMT`,
                        `SALES AMT`,
                        `COLLECTION`,
                        `STATUS`,
                        `CONTRACT DATE`
                    FROM osstem_collection
                    WHERE DATE_FORMAT(`DATE`, '%Y-%m') = '{year_month}'
                """
            }

            with self.db_manager.get_connection() as conn:
                # 分批读取销售数据
                sales_chunks = []
                for chunk in pd.read_sql(text(queries['sales']), conn, chunksize=10000):
                    chunk['CUSTOMER'] = chunk['CUSTOMER'].astype(str)
                    sales_chunks.append(chunk)
                sales_df = pd.concat(sales_chunks, ignore_index=True) if sales_chunks else pd.DataFrame()

                # 读取客户主数据
                collection_df = pd.read_sql(text(queries['collection']), conn)

                # 分批读取合同数据
                contract_chunks = []
                for chunk in pd.read_sql(text(queries['contract']), conn, chunksize=10000):
                    chunk['CUSTOMER'] = chunk['CUSTOMER'].astype(str)
                    contract_chunks.append(chunk)
                contract_df = pd.concat(contract_chunks, ignore_index=True) if contract_chunks else pd.DataFrame()

                self.logger.info(f"成功加载 {year}年{month}月 的数据")
                self.logger.debug(
                    f"加载数据统计: 销售记录 {len(sales_df)}条, 客户记录 {len(collection_df)}条, 合同记录 {len(contract_df)}条")

                return sales_df, collection_df, contract_df

        except Exception as e:
            self.logger.error(f"加载 {year}年{month}月 数据时发生错误: {str(e)}")
            raise DatabaseError(f"月度数据加载失败: {str(e)}")

    def get_active_customers(self, year: int, month: int) -> List[str]:
        """获取指定月份的活跃客户列表

        参数:
            year: 年份
            month: 月份

        返回:
            list: 活跃客户编号列表
        """
        query = f"""
            SELECT DISTINCT CAST(CUSTOMER as CHAR) as CUSTOMER 
            FROM (
                -- 获取有销售记录的客户
                SELECT DISTINCT `CST CODE` AS CUSTOMER 
                FROM osstem_sales_list
                WHERE YEAR(`SALES DATE`) = {year}
                AND MONTH(`SALES DATE`) = {month}

                UNION

                -- 获取有合同记录的客户
                SELECT DISTINCT CUSTOMER 
                FROM osstem_collection
                WHERE YEAR(`DATE`) = {year}
                AND MONTH(`DATE`) = {month}
            ) a
            WHERE CUSTOMER IS NOT NULL
        """

        try:
            with self.db_manager.get_connection() as conn:
                df = pd.read_sql(text(query), conn)
                customers = df['CUSTOMER'].tolist()
                self.logger.info(f"找到 {year}年{month}月 的 {len(customers)} 个活跃客户")
                return customers
        except Exception as e:
            self.logger.error(f"获取活跃客户列表时发生错误: {str(e)}")
            raise DatabaseError("获取活跃客户列表失败")


class ConfigLoader:
    """配置加载器

    负责从数据库加载和缓存评分系统的配置参数
    包括评分标准、权重、客户类型等配置信息
    """

    def __init__(self, db_manager: DatabaseManager):
        """初始化配置加载器

        参数:
            db_manager: 数据库管理器实例
        """
        self.db_manager = db_manager
        self.config_cache = {}  # 配置缓存字典
        self.last_load_time = None  # 最后加载时间
        self.cache_duration = timedelta(minutes=30)  # 缓存有效期（30分钟）

    def get_config(self, config_type: str, reload: bool = False) -> Dict:
        """获取指定类型的配置

        参数:
            config_type: 配置类型（如：'contract_amount', 'customer_type'等）
            reload: 是否强制重新加载配置

        返回:
            dict: 配置数据字典
        """
        current_time = datetime.now()
        cache_key = f"{config_type}"

        # 检查是否需要重新加载配置
        needs_reload = (
                reload or
                cache_key not in self.config_cache or
                self.last_load_time is None or
                current_time - self.last_load_time > self.cache_duration
        )

        if needs_reload:
            query = """
                SELECT config_key, config_value
                FROM credit_score_config
                WHERE config_type = :config_type
                    AND status = 1  -- 仅获取有效配置
                    AND effective_date <= CURRENT_DATE  -- 仅获取已生效配置
                ORDER BY effective_date DESC  -- 按生效日期降序排序，获取最新配置
            """

            try:
                with self.db_manager.get_connection() as conn:
                    result = conn.execute(text(query), {"config_type": config_type})
                    # 解析配置值（JSON字符串转为Python对象）
                    config_data = {
                        row[0]: json.loads(row[1]) if row[1].startswith('{') else row[1]
                        for row in result
                    }

                if not config_data:
                    raise ConfigurationError(f"未找到类型为 {config_type} 的有效配置")

                self.config_cache[cache_key] = config_data
                self.last_load_time = current_time
                logger.debug(f"已加载配置类型: {config_type}")

            except json.JSONDecodeError as e:
                logger.error(f"配置值JSON解析失败: {str(e)}")
                raise ConfigurationError(f"配置值格式错误: {str(e)}")
            except Exception as e:
                logger.error(f"加载配置 {config_type} 时发生错误: {str(e)}")
                raise ConfigurationError(f"配置加载失败: {str(e)}")

        return self.config_cache[cache_key]

    def validate_config(self) -> bool:
        """验证配置完整性"""
        required_configs = [
            'contract_amount',  # 合同金额评分标准
            'customer_type',  # 客户类型评分标准
            'weight',  # 各项评分权重
            'credit_level'  # 信用等级划分标准
        ]

        try:
            for config_type in required_configs:
                config_data = self.get_config(config_type)
                if not config_data:
                    logger.error(f"缺少必需的配置项: {config_type}")
                    return False
            logger.info("配置完整性验证通过")
            return True

        except Exception as e:
            logger.error(f"配置验证过程中发生错误: {str(e)}")
            return False

    def refresh_all_configs(self):
        """刷新所有配置缓存"""
        try:
            self.config_cache.clear()
            self.last_load_time = None
            logger.info("配置缓存已清空，将在下次访问时重新加载")
        except Exception as e:
            logger.error(f"刷新配置缓存时发生错误: {str(e)}")
            raise ConfigurationError("配置缓存刷新失败")

    def get_config_history(self, config_type: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取配置历史记录

        参数:
            config_type: 配置类型
            start_date: 开始日期
            end_date: 结束日期

        返回:
            DataFrame: 配置历史记录数据框
        """
        query = """
            SELECT 
                config_key,
                config_value,
                effective_date,
                created_at
            FROM credit_score_config
            WHERE config_type = :config_type
                AND effective_date BETWEEN :start_date AND :end_date
            ORDER BY effective_date DESC, created_at DESC
        """

        try:
            with self.db_manager.get_connection() as conn:
                df = pd.read_sql(
                    query,
                    conn,
                    params={
                        "config_type": config_type,
                        "start_date": start_date,
                        "end_date": end_date
                    }
                )
                logger.info(f"成功获取配置 {config_type} 的历史记录")
                return df

        except Exception as e:
            logger.error(f"获取配置历史记录时发生错误: {str(e)}")
            raise DatabaseError("获取配置历史记录失败")


class ScoreLogger:
    """评分日志记录器

    负责记录评分变更和系统告警信息
    """

    def __init__(self, db_manager: DatabaseManager):
        """初始化日志记录器

        参数:
            db_manager: 数据库管理器实例
        """
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        self.setup_logging()

    def setup_logging(self):
        """设置日志记录系统"""
        self.logger.setLevel(logging.INFO)

        # 文件处理器
        fh = logging.FileHandler('logs/credit_score.log', encoding='utf-8')
        fh.setLevel(logging.INFO)

        # 控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # 日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # 添加处理器（避免重复添加）
        if not self.logger.handlers:
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

    def log_score_change(self, customer_code: str, year: int, month: int,
                         old_score: float, new_score: float, reason: str):
        """记录评分变更"""
        query = """
            INSERT INTO credit_score_history (
                customer_code, score_year, score_month,
                old_score, new_score, change_reason
            ) VALUES (
                :customer_code, :year, :month,
                :old_score, :new_score, :reason
            )
        """
        try:
            with self.db_manager.get_connection() as conn:
                conn.execute(text(query), {
                    "customer_code": customer_code,
                    "year": year,
                    "month": month,
                    "old_score": old_score,
                    "new_score": new_score,
                    "reason": reason
                })
                conn.commit()  # 确保提交事务
            self.logger.info(f"客户 {customer_code} 的评分变更已记录: {old_score} -> {new_score}")
        except Exception as e:
            self.logger.error(f"记录评分变更时发生错误: {str(e)}")
            raise DatabaseError("评分变更记录失败")

    def log_alert(self, customer_code: str, alert_type: str,
                  alert_level: str, content: str):
        """记录告警信息"""
        query = """
            INSERT INTO credit_score_alert (
                customer_code, alert_type, alert_level, alert_content
            ) VALUES (
                :customer_code, :alert_type, :alert_level, :content
            )
        """
        try:
            with self.db_manager.get_connection() as conn:
                conn.execute(text(query), {
                    "customer_code": customer_code,
                    "alert_type": alert_type,
                    "alert_level": alert_level,
                    "content": content
                })
                conn.commit()  # 确保提交事务
            self.logger.info(f"已记录客户 {customer_code} 的{alert_level}级告警: {alert_type}")
        except Exception as e:
            self.logger.error(f"记录告警信息时发生错误: {str(e)}")
            raise DatabaseError("告警信息记录失败")

    def get_recent_alerts(self, days: int = 7) -> pd.DataFrame:
        """获取最近的告警记录

        参数:
            days: 查询天数（默认7天）

        返回:
            DataFrame: 告警记录数据框
        """
        query = """
            SELECT *
            FROM credit_score_alert
            WHERE created_at >= DATE_SUB(CURRENT_DATE, INTERVAL :days DAY)
            ORDER BY created_at DESC
        """

        try:
            with self.db_manager.get_connection() as conn:
                df = pd.read_sql(text(query), conn, params={"days": days})
                return df
        except Exception as e:
            self.logger.error(f"获取告警记录时发生错误: {str(e)}")
            raise DatabaseError("获取告警记录失败")

    def log_validation_result(self, customer_code: str, validation_result: Dict) -> None:
        """记录验证结果"""
        try:
            success = validation_result.get('success', False)
            log_level = logging.INFO if success else logging.WARNING

            self.logger.log(log_level, f"客户 {customer_code} 验证结果:")
            if not success:
                for error in validation_result.get('errors', []):
                    self.logger.warning(f"  错误: {error}")
            for warning in validation_result.get('warnings', []):
                self.logger.warning(f"  警告: {warning}")

        except Exception as e:
            self.logger.error(f"记录验证结果时发生错误: {str(e)}")


class CreditScoreUtils:
    """评分系统工具类

    提供评分系统所需的各种通用工具函数
    包括日期处理、数据验证、格式转换等功能
    """

    @staticmethod
    def validate_date(year: int, month: int) -> bool:
        """验证日期有效性

        参数:
            year: 年份
            month: 月份

        返回:
            bool: 日期是否有效
        """
        try:
            datetime(year, month, 1)
            return True
        except ValueError:
            logger.error(f"无效的日期: {year}年{month}月")
            return False

    @staticmethod
    def format_year_month(year: int, month: int) -> str:
        """格式化年月字符串

        参数:
            year: 年份
            month: 月份

        返回:
            str: 格式化的年月字符串 (YYYY-MM)
        """
        return f"{year:04d}-{month:02d}"

    @staticmethod
    def calculate_date_range(year: int, month: int) -> Tuple[str, str]:
        """计算指定月份的起止日期

        参数:
            year: 年份
            month: 月份

        返回:
            tuple: (月初日期, 月末日期)的元组
        """
        start_date = f"{year}-{month:02d}-01"
        end_date = (pd.to_datetime(start_date) + MonthEnd(0)).strftime('%Y-%m-%d')
        logger.debug(f"计算日期范围: {start_date} 至 {end_date}")
        return start_date, end_date

    @staticmethod
    def format_customer_code(customer_code: Any) -> str:
        """格式化客户编号

        参数:
            customer_code: 任意类型的客户编号

        返回:
            str: 格式化后的客户编号字符串
        """
        return str(customer_code).strip()

    @staticmethod
    def validate_score(score: float) -> float:
        """验证并规范化评分值

        确保评分在0-100的范围内

        参数:
            score: 原始评分值

        返回:
            float: 规范化后的评分值

        异常:
            DataValidationError: 评分值无效时抛出
        """
        try:
            score_float = float(score)
            normalized_score = max(0, min(100, score_float))
            if normalized_score != score_float:
                logger.warning(f"评分值已被规范化: {score_float} -> {normalized_score}")
            return normalized_score
        except (ValueError, TypeError):
            error_msg = f"无效的评分值: {score}"
            logger.error(error_msg)
            raise DataValidationError(error_msg)

    @staticmethod
    def calculate_percentage(value: float, total: float) -> float:
        """计算百分比

        参数:
            value: 分子值
            total: 分母值

        返回:
            float: 百分比值（0-100）
        """
        try:
            if total == 0:
                logger.warning("计算百分比时分母为0")
                return 0.0
            percentage = (value / total) * 100
            return percentage
        except (ValueError, TypeError):
            logger.error(f"计算百分比时发生错误: value={value}, total={total}")
            return 0.0

    @staticmethod
    def format_currency(amount: float) -> str:
        """格式化货币金额

        参数:
            amount: 金额数值

        返回:
            str: 格式化后的金额字符串
        """
        try:
            return f"¥{amount:,.2f}"
        except (ValueError, TypeError):
            logger.error(f"金额格式化失败: {amount}")
            return "¥0.00"

    @staticmethod
    def calculate_growth_rate(current: float, previous: float) -> float:
        """计算增长率

        参数:
            current: 当前值
            previous: 上期值

        返回:
            float: 增长率（小数形式）
        """
        try:
            if previous == 0:
                logger.warning("计算增长率时基期值为0")
                return 0.0
            growth_rate = (current - previous) / previous
            return growth_rate
        except (ValueError, TypeError):
            logger.error(f"计算增长率时发生错误: current={current}, previous={previous}")
            return 0.0

    @staticmethod
    def check_value_range(value: float, min_value: float, max_value: float) -> bool:
        """检查数值是否在指定范围内

        参数:
            value: 待检查的值
            min_value: 最小值
            max_value: 最大值

        返回:
            bool: 是否在范围内
        """
        try:
            if not (min_value <= value <= max_value):
                logger.warning(f"数值 {value} 不在范围 [{min_value}, {max_value}] 内")
                return False
            return True
        except TypeError:
            logger.error(f"数值范围检查失败: value={value}, range=[{min_value}, {max_value}]")
            return False

    @classmethod
    def generate_score_summary(cls, scores: List[float]) -> Dict[str, float]:
        """生成评分统计摘要

        参数:
            scores: 评分列表

        返回:
            dict: 包含均值、中位数、最大值、最小值的字典
        """
        try:
            if not scores:
                logger.warning("评分列表为空")
                return {
                    "mean": 0.0,
                    "median": 0.0,
                    "max": 0.0,
                    "min": 0.0
                }

            return {
                "mean": float(np.mean(scores)),
                "median": float(np.median(scores)),
                "max": float(max(scores)),
                "min": float(min(scores))
            }
        except Exception as e:
            logger.error(f"生成评分摘要时发生错误: {str(e)}")
            raise DataValidationError(f"评分摘要生成失败: {str(e)}")