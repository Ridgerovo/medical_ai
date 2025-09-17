from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import uuid
from app.db.session import get_db
from app.models.doctor import Doctor, GenderEnum
from app.schemas.doctor import DoctorRegister, DoctorLogin, DoctorResponse

# 创建路由实例（前缀为/doctor，标签为"医生管理"）
router = APIRouter(
    prefix="/doctor",
    tags=["医生管理"]
)


@router.post("/register", response_model=DoctorResponse, status_code=status.HTTP_201_CREATED)
def register_doctor(
        doctor_info: DoctorRegister,
        db: Session = Depends(get_db)
):
    """医生注册接口"""
    # 1. 检查工号是否已注册
    existing_doctor = db.query(Doctor).filter(
        Doctor.employee_id == doctor_info.employee_id
    ).first()
    if existing_doctor:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"工号 {doctor_info.employee_id} 已被注册"
        )

    # 2. 生成唯一doctor_id（格式：DOC-随机12位大写字母数字）
    doctor_id = f"DOC-{uuid.uuid4().hex[:12].upper()}"

    # 3. 创建医生记录
    new_doctor = Doctor(
        doctor_id=doctor_id,
        employee_id=doctor_info.employee_id,
        password=doctor_info.password,  # 注意：实际项目需加密存储
        name=doctor_info.name,
        gender=GenderEnum(doctor_info.gender),  # 转换为枚举类型
        age=doctor_info.age,
        is_admin=doctor_info.is_admin
    )

    # 4. 保存到数据库
    db.add(new_doctor)
    db.commit()
    db.refresh(new_doctor)  # 刷新数据，获取数据库自动生成的字段（如registration_time）

    return new_doctor


@router.post("/login")
def login_doctor(
        login_info: DoctorLogin,
        db: Session = Depends(get_db)
):
    """医生登录接口"""
    # 1. 查询医生信息
    doctor = db.query(Doctor).filter(
        Doctor.employee_id == login_info.employee_id
    ).first()

    # 2. 验证医生是否存在及密码是否正确
    if not doctor or doctor.password != login_info.password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="工号或密码错误"
        )

    # 3. 返回登录成功信息（实际项目中应返回Token）
    return {
        "status": "success",
        "message": "登录成功",
        "data": {
            "doctor_id": doctor.doctor_id,
            "employee_id": doctor.employee_id,
            "name": doctor.name,
            "is_admin": doctor.is_admin
        }
    }