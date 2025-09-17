import os
from dotenv import load_dotenv

# 加载环境变量（可选，用于存放敏感信息）
load_dotenv()

# 数据库配置
DATABASE_CONFIG = {
    "url": os.getenv("DATABASE_URL", "mysql+pymysql://root:123456@localhost:3306/medical_ai?charset=utf8mb4"),
    "echo": False  # 是否打印SQL日志（开发阶段可设为True）
}

# JWT配置（用于身份验证）
JWT_SECRET_KEY = "your-secret-key-here"  # 实际生产环境中应使用更复杂的密钥
JWT_ALGORITHM = "HS256"  # 加密算法
ACCESS_TOKEN_EXPIRE_MINUTES = 30  # 令牌有效期（分钟）


# API配置
API_PREFIX = "/api/v1"

# 新增影像存储配置
IMAGE_STORAGE = {
    "local_root": "data/local_images",
    "pacs_root": "data/pacs_images",
    "default_pacs_url": "http://pacs-server:8080/wado"
}