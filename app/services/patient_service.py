from app.db.session import SessionLocal
from app.schemas.patient import PatientCreate, PatientNameQuery
from app.models.patient import Patient
from sqlalchemy.orm import Session
from typing import Optional, List


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# 新增患者
def create_patient(patient_create: PatientCreate, db: Session) -> Patient:
    """创建新患者记录"""
    patient = Patient(
        name=patient_create.name,
        gender=patient_create.gender,
        age=patient_create.age,
        phone=patient_create.phone
    )

    db.add(patient)
    db.commit()
    db.refresh(patient)
    return patient


# 删除患者
def delete_patient(patient_id: str, db: Session) -> bool:
    """根据ID删除患者记录"""
    patient = db.query(Patient).filter(Patient.id == patient_id).first()

    if not patient:
        return False

    db.delete(patient)
    db.commit()
    return True


# 根据ID获取患者
def get_patient_by_id(patient_id: str, db: Session) -> Optional[Patient]:
    """根据ID获取患者详情"""
    return db.query(Patient).filter(Patient.id == patient_id).first()


# 按姓名查询患者（模糊匹配）
def search_patients_by_name(
        query: PatientNameQuery,
        db: Session,
        skip: int = 0,
        limit: int = 20
) -> List[Patient]:
    """根据姓名模糊查询患者"""
    return db.query(Patient)\
        .filter(Patient.name.like(f"%{query.name}%"))\
        .order_by(Patient.create_time.desc())\
        .offset(skip)\
        .limit(limit)\
        .all()


# 获取同名患者总数
def get_patients_count_by_name(query: PatientNameQuery, db: Session) -> int:
    """获取符合姓名查询条件的患者总数"""
    return db.query(Patient)\
        .filter(Patient.name.like(f"%{query.name}%"))\
        .count()

def get_all_patients(db: Session, skip: int = 0, limit: int = 10) -> List[Patient]:
    """查询所有患者（支持分页）"""
    return db.query(Patient)\
        .order_by(Patient.create_time.desc())\
        .offset(skip)\
        .limit(limit)\
        .all()