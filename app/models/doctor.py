from sqlalchemy import Column, String, Integer, Enum, DateTime
from sqlalchemy.sql import func
from app.db.base import Base
import enum


# 性别枚举（限制只能输入男/女）
class GenderEnum(str, enum.Enum):
    MALE = "男"
    FEMALE = "女"


class Doctor(Base):
    """医生信息表模型"""
    __tablename__ = "doctor"  # 数据库表名

    # 主键：医生唯一ID（注册时自动生成）
    doctor_id = Column(
        String(32),
        primary_key=True,
        comment="医生唯一标识ID"
    )

    # 工号（登录账号，8位数字，唯一）
    employee_id = Column(
        String(8),
        unique=True,
        nullable=False,
        comment="8位数字工号（登录账号）"
    )

    # 登录密码（实际项目需加密存储）
    password = Column(
        String(100),
        nullable=False,
        comment="登录密码"
    )

    # 医生姓名
    name = Column(
        String(50),
        nullable=False,
        comment="医生姓名"
    )

    # 性别（关联GenderEnum）
    gender = Column(
        Enum(GenderEnum),
        nullable=False,
        comment="性别"
    )

    # 年龄
    age = Column(
        Integer,
        nullable=False,
        comment="年龄"
    )

    # 管理员标识（0=普通医生，1=管理员）
    is_admin = Column(
        Integer,
        default=0,
        comment="是否为管理员：0-否，1-是"
    )

    # 注册时间（自动记录）
    registration_time = Column(
        DateTime,
        server_default=func.now(),
        comment="注册时间"
    )