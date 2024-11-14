"""
客户信用评分系统
~~~~~~~~~~~~~~~~~~~

该系统用于计算和管理牙科医疗器械行业客户的信用评分。

基本用法:
    >>> from credit_score_system import CreditScoreSystem
    >>> system = CreditScoreSystem(config)
    >>> score = system.process_single_customer("4000208", 2023, 12)
"""

__version__ = '1.0.0'
__author__ = '司海彭'
__license__ = 'Private'

# 版本信息元组
VERSION = (1, 0, 0)

# 设置默认日志配置
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# 按正确的依赖顺序导入
from .utils import (
    DatabaseConfig,
    DatabaseManager,
    DataLoader,
    ConfigLoader,
    ScoreLogger,
    CreditScoreUtils,
    CreditScoreException,
    DataValidationError,
    ScoreCalculationError,
    DatabaseError,
    ConfigurationError
)

from .validation import DataValidator
from .calculator import CreditScoreCalculator
from .core import CreditScoreSystem

def get_version():
    """获取版本号"""
    return '.'.join(str(v) for v in VERSION)

# 导出所有应该公开的内容
__all__ = [
    'CreditScoreSystem',
    'DatabaseConfig',
    'DatabaseManager',
    'DataLoader',
    'ConfigLoader',
    'ScoreLogger',
    'CreditScoreUtils',
    'CreditScoreCalculator',
    'DataValidator',
    'CreditScoreException',
    'DataValidationError',
    'ScoreCalculationError',
    'DatabaseError',
    'ConfigurationError',
    'get_version'
]