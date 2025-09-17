from sqlalchemy import Column, String, DateTime, ForeignKey, func
from sqlalchemy.orm import relationship
from app.db.base import Base


class MedicalImage(Base):
    __tablename__ = "medical_image"  # 数据库表名

    # 影像唯一ID
    image_id = Column(
        String(32),
        primary_key=True,
        comment="影像唯一标识ID"
    )

    # 关联患者ID（外键，关联t_patient表的id字段）
    patient_id = Column(
        String(36),  # 患者表id是36位UUID，这里需保持长度一致
        ForeignKey("t_patient.id"),  # 患者表名是t_patient，主键是id
        nullable=False,
        comment="关联患者ID（对应t_patient表的id）"
    )

    # 患者姓名（冗余存储，来自t_patient表的name字段）
    patient_name = Column(
        String(50),
        nullable=False,
        comment="患者姓名"
    )

    # 影像存储路径
    file_path = Column(
        String(255),
        nullable=False,
        comment="影像文件存储路径"
    )

    # 影像文件名
    file_name = Column(
        String(100),
        nullable=False,
        comment="影像文件名（PNG格式）"
    )

    # 影像类型（固定为PNG）
    image_type = Column(
        String(10),
        default="PNG",
        nullable=False,
        comment="影像类型（仅支持PNG）"
    )

    # 导入医生ID（外键，关联doctor表的doctor_id字段）
    doctor_id = Column(
        String(32),
        ForeignKey("doctor.doctor_id"),  # 医生表名是doctor，主键是doctor_id
        nullable=False,
        comment="导入医生ID（对应doctor表的doctor_id）"
    )

    # 导入医生姓名（冗余存储，来自doctor表的name字段）
    doctor_name = Column(
        String(50),
        nullable=False,
        comment="导入医生姓名"
    )

    # 导入时间（自动生成）
    import_time = Column(
        DateTime,
        server_default=func.now(),
        comment="影像导入时间"
    )

    # 关联关系：影像-医生（一个医生可导入多个影像）
    doctor = relationship("Doctor", backref="imported_images")

    # 关联关系：影像-患者（一个患者可有多张影像）
    patient = relationship("Patient", backref="medical_images")
