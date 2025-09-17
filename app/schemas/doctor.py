from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime

class DoctorRegister(BaseModel):
    """医生注册请求模型（接口输入参数校验）"""
    employee_id: str = Field(..., min_length=8, max_length=8, description="8位数字工号")
    password: str = Field(..., min_length=6, description="登录密码（至少6位）")
    name: str = Field(..., min_length=1, max_length=50, description="医生姓名")
    gender: str = Field(..., description="性别（男/女）")
    age: int = Field(..., ge=22, le=65, description="年龄（22-65岁）")
    is_admin: Optional[int] = Field(0, ge=0, le=1, description="是否为管理员：0-否，1-是")

    # 校验工号必须是数字
    @field_validator('employee_id')
    def employee_id_must_be_digit(cls, v):
        if not v.isdigit():
            raise ValueError('工号必须是8位数字')
        return v

    # 校验性别必须是男或女
    @field_validator('gender')
    def gender_must_be_valid(cls, v):
        if v not in ['男', '女']:
            raise ValueError('性别只能是"男"或"女"')
        return v

class DoctorLogin(BaseModel):
    """医生登录请求模型"""
    employee_id: str = Field(..., min_length=8, max_length=8, description="8位数字工号")
    password: str = Field(..., description="登录密码")

class DoctorResponse(BaseModel):
    """医生信息响应模型（接口返回数据格式）"""
    doctor_id: str
    employee_id: str
    name: str
    gender: str
    age: int
    is_admin: int
    registration_time: datetime

    # 支持从ORM模型直接转换
    class Config:
        from_attributes = True