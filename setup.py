# setup.py
import os
import sys
from pathlib import Path
import json
from datetime import datetime
import logging

# 设置日志
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f"setup_{datetime.now().strftime('%Y%m%d')}.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def main():
    try:
        # 添加项目根目录到Python路径
        project_root = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, project_root)

        # 加载配置
        config_path = os.path.join(project_root, 'config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 导入系统
        from credit_score_system import CreditScoreSystem

        # 初始化系统
        system = CreditScoreSystem(config)

        # 处理历史数据
        start_year = 2016
        end_year = 2023

        logger.info(f"开始处理 {start_year}-{end_year} 年度数据")

        results = system.process_history(start_year, end_year)

        logger.info("历史数据处理完成")

    except Exception as e:
        logger.error(f"运行时发生错误: {str(e)}")
        raise


if __name__ == "__main__":
    main()