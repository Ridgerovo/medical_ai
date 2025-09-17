from pydantic import BaseModel, Field, validator
from typing import Optional
from datetime import datetime
import enum


class GenderEnum(str, enum.Enum):
    MALE = "男"
    FEMALE = "女"


# 患者创建请求模型（保持不变）
class PatientCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=50, description="患者姓名")
    gender: GenderEnum = Field(..., description="性别")
    age: int = Field(..., ge=0, le=120, description="年龄，0-120之间")
    phone: Optional[str] = Field(None, description="联系电话")

    @validator('phone')
    def validate_phone(cls, v):
        if v and (len(v) != 11 or not v.isdigit()):
            raise ValueError('联系电话必须是11位数字')
        return v

# 患者姓名查询模型
class PatientNameQuery(BaseModel):
    name: str = Field(..., description="患者姓名（模糊查询）")

# 患者响应模型（保持不变）
class PatientOut(BaseModel):
    id: str
    name: str
    gender: GenderEnum
    age: int
    phone: Optional[str]
    create_time: datetime

    class Config:
        orm_mode = True  # 支持从ORM模型直接转换
