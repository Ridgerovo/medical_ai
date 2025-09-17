from app.db.base import Base
from sqlalchemy import Column, Integer, String, DateTime, Enum
from datetime import datetime
import uuid
import enum


class GenderEnum(str, enum.Enum):
    MALE = "男"
    FEMALE = "女"


class Patient(Base):
    __tablename__ = "t_patient"

    # 患者ID：使用UUID作为主键，创建时自动生成
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), comment="患者ID（UUID）")
    name = Column(String(50), nullable=False, comment="患者姓名")
    gender = Column(Enum(GenderEnum), nullable=False, comment="性别")
    age = Column(Integer, nullable=False, comment="年龄")
    phone = Column(String(11), comment="联系电话")
    create_time = Column(DateTime, default=datetime.now, comment="创建时间")

    def __repr__(self):
        return f"<Patient(id='{self.id}', name='{self.name}')>"
