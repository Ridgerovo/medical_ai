from sqlalchemy import Column, String, DateTime, Text, ForeignKey, func, Integer
from sqlalchemy.orm import relationship
from app.db.base import Base
import uuid


class Report(Base):
    __tablename__ = "medical_report"  # 数据库表名

    # 报告唯一ID（主键）
    report_id = Column(
        String(32),
        primary_key=True,
        default=lambda: f"REP-{uuid.uuid4().hex[:12].upper()}",
        comment="报告唯一标识ID"
    )

    # 关联医生ID（外键，关联doctor表）
    doctor_id = Column(
        String(32),
        ForeignKey("doctor.doctor_id"),
        nullable=False,
        comment="生成报告的医生ID"
    )

    # 医生姓名（冗余存储）
    doctor_name = Column(
        String(50),
        nullable=False,
        comment="医生姓名"
    )

    # 患者姓名（直接存储，无患者ID关联）
    patient_name = Column(
        String(50),
        nullable=False,
        comment="患者姓名"
    )

    # 创建日期（自动生成）
    create_time = Column(
        DateTime,
        server_default=func.now(),
        comment="报告创建时间"
    )

    # 原始图像路径（关联影像表的file_path）
    original_image_path = Column(
        String(255),
        nullable=False,
        comment="原始影像文件路径"
    )

    # 分析后图像路径（AI处理后的图像）
    analyzed_image_path = Column(
        String(255),
        nullable=False,
        comment="分析后影像文件路径"
    )

    # 报告描述（诊断结论等）
    description = Column(
        Text,
        nullable=False,
        comment="报告详细描述"
    )

    # 报告类型
    report_type = Column(
        String(50),
        nullable=False,
        comment="报告类型（如：常规检查、CT分析、MRI诊断等）"
    )

    # 报告状态（0-正常，1-审核中，2-已归档），默认0
    report_status = Column(
        Integer,
        nullable=False,
        default=0,
        comment="报告状态（0:正常, 1:审核中, 2:已归档）"
    )

    # 关联关系：报告-医生
    doctor = relationship("Doctor", backref="reports")

    # 移除与患者表的外键关联（因已删除patient_id）