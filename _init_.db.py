from app.db.base import Base
from app.db.session import engine
from app.models.doctor import Doctor
from app.models.image import MedicalImage
from app.models.patient import Patient
from app.models.report import Report

def init_database():
    """创建所有表（如果不存在）"""
    Base.metadata.create_all(bind=engine)
    print("✅ 数据库表初始化完成！")

if __name__ == "__main__":
    init_database()