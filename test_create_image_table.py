from sqlalchemy import create_engine
from app.db.base import Base
from app.models.image import MedicalImage  # 直接导入影像模型
from config import DATABASE_CONFIG

# 1. 打印模型元数据（验证模型是否被正确识别）
print("影像模型表名:", MedicalImage.__tablename__)  # 应输出 'medical_image'
print("影像模型继承的Base:", MedicalImage.__base__ is Base)  # 必须为 True

# 2. 连接数据库并尝试创建表
try:
    # 创建引擎（使用配置中的URL）
    engine = create_engine(DATABASE_CONFIG["url"])

    # 只创建MedicalImage表（排除其他表的干扰）
    MedicalImage.__table__.create(bind=engine, checkfirst=True)
    print("✅ medical_image表创建/验证成功！")

    # 3. 验证表是否真的存在
    with engine.connect() as conn:
        result = conn.execute("SHOW TABLES LIKE 'medical_image'")
        if result.fetchone():
            print("📊 数据库中已存在medical_image表")
        else:
            print("❌ 数据库中仍不存在medical_image表（连接的数据库可能不对）")

except Exception as e:
    print(f"❌ 执行失败：{e}")