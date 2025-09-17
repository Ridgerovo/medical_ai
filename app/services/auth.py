from datetime import datetime, timedelta
from typing import Optional, Annotated
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

# 先导入API_PREFIX，确保在使用前已定义
from config import API_PREFIX

# 导入其他配置
from config import JWT_SECRET_KEY, JWT_ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES
# 导入数据库会话和医生模型
from app.db.session import get_db
from app.models.doctor import Doctor

# 密码加密上下文
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# 现在可以正确使用API_PREFIX了
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{API_PREFIX}/doctor/login")  # 登录接口地址


# 以下代码保持不变...
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证明文密码与加密密码是否匹配"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """对密码进行加密"""
    return pwd_context.hash(password)


def get_doctor_by_employee_id(db: Session, employee_id: str) -> Optional[Doctor]:
    """通过工号查询医生"""
    return db.query(Doctor).filter(Doctor.employee_id == employee_id).first()


def authenticate_doctor(db: Session, employee_id: str, password: str) -> Optional[Doctor]:
    """验证医生身份（用于登录）"""
    doctor = get_doctor_by_employee_id(db, employee_id)
    if not doctor:
        return None
    if not verify_password(password, doctor.password):
        return None
    return doctor


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """生成JWT访问令牌"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt


async def get_current_doctor(
        token: Annotated[str, Depends(oauth2_scheme)],
        db: Annotated[Session, Depends(get_db)]
) -> Doctor:
    """获取当前登录的医生信息（用于接口权限控制）"""
    # 验证失败的异常
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无法验证凭据",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        # 解码JWT令牌
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        employee_id: str = payload.get("sub")  # 从payload中获取工号
        if employee_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    # 查询医生信息
    doctor = get_doctor_by_employee_id(db, employee_id=employee_id)
    if doctor is None:
        raise credentials_exception
    return doctor