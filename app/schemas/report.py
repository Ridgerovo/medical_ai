# app/schemas/report.py
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class ReportCreate(BaseModel):
    doctor_id: str
    doctor_name: str
    patient_name: str
    original_image_path: str
    analyzed_image_path: str
    description: str
    report_type: str = Field(..., description="报告类型（如：CT/MRI）")
    report_status: int = Field(0, ge=0, le=2, description="报告状态（0-正常，1-审核中，2-已归档）")


class ReportCreateWithFiles(BaseModel):
    doctor_id: str = Field(..., description="医生ID")
    doctor_name: str = Field(..., description="医生姓名")
    patient_name: str = Field(..., description="患者姓名")
    description: str = Field(..., description="报告描述")
    report_type: str = Field(..., description="报告类型（如：CT/MRI）")
    report_status: int = Field(0, ge=0, le=2, description="报告状态（0-正常，1-审核中，2-已归档）")


# 新增：报告更新模型（仅包含允许修改的字段）
class ReportUpdate(BaseModel):
    description: Optional[str] = Field(None, description="诊断结论描述（可选更新）")
    report_status: Optional[int] = Field(
        None,
        ge=0,
        le=2,
        description="报告状态（0-正常，1-审核中，2-已归档，可选更新）"
    )


class ReportResponse(BaseModel):
    report_id: str
    doctor_id: str
    doctor_name: str
    patient_name: str
    create_time: datetime
    original_image_path: str
    analyzed_image_path: str
    description: str
    report_type: str
    report_status: int

    class Config:
        from_attributes = True  # 支持ORM模型转换