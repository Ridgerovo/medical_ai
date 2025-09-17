from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typing import Generator
from config import DATABASE_CONFIG
from app.db.base import Base

# 创建数据库引擎
engine = create_engine(
    DATABASE_CONFIG["url"],
    echo=DATABASE_CONFIG["echo"]  # 控制是否打印SQL日志
)

# 会话工厂（用于创建数据库会话）
SessionLocal = sessionmaker(
    autocommit=False,  # 关闭自动提交
    autoflush=False,   # 关闭自动刷新
    bind=engine
)

def get_db() -> Generator:
    """
    数据库会话依赖项
    为每个请求创建独立会话，请求结束后自动关闭
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()