from .session import get_db  # 关键：将session.py中的get_db导入到app.db包

# 可选：导出数据库引擎和会话工厂（如果其他地方需要直接使用）
from .session import engine, SessionLocal