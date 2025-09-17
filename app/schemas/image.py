from pydantic import BaseModel, Field, validator
from typing import Optional
from datetime import datetime

class ImageCreate(BaseModel):
    """影像新增请求模型（适配文件上传，后端自动保存）"""
    # 必选：患者ID（关联t_patient表）
    patient_id: str = Field(..., description="关联患者ID（必填）")
    # 必选：医生ID（关联医生表）
    doctor_id: str = Field(..., description="导入医生ID（必填）")
    # 文件名由后端从上传文件中提取，前端无需传（删除原file_name字段）

class ImageResponse(BaseModel):
    """影像响应模型（包含后端生成的文件路径）"""
    image_id: str
    file_name: str  # 后端生成的唯一文件名
    file_path: str  # 后端自动保存的项目内路径
    image_type: str = "PNG"  # 固定为PNG
    patient_id: str
    patient_name: str
    doctor_id: str
    doctor_name: str
    import_time: datetime

    class Config:
        from_attributes = True