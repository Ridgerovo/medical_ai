from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List
from app.models.patient import Patient

from app.schemas.patient import PatientCreate, PatientOut, PatientNameQuery
from app.services.patient_service import (
    create_patient,
    delete_patient,
    get_patient_by_id,
    search_patients_by_name,
    get_patients_count_by_name,
    get_all_patients,
    get_db
)

router = APIRouter(
    prefix="/patients",
    tags=["patients"]
)


@router.post("/", response_model=PatientOut, status_code=201)
def add_patient(
        patient: PatientCreate,
        db: Session = Depends(get_db)
):
    """新增患者"""
    return create_patient(patient, db)


@router.delete("/{patient_id}", status_code=200)
def remove_patient(
        patient_id: str,
        db: Session = Depends(get_db)
):
    """删除患者"""
    success = delete_patient(patient_id, db)
    if not success:
        raise HTTPException(status_code=404, detail="患者不存在")
    return {"message": "患者删除成功"}


@router.get("/{patient_id}", response_model=PatientOut)
def get_patient(
        patient_id: str,
        db: Session = Depends(get_db)
):
    """通过ID查询患者"""
    patient = get_patient_by_id(patient_id, db)
    if not patient:
        raise HTTPException(status_code=404, detail="患者不存在")
    return patient


@router.get("/search/by-name", response_model=List[PatientOut])
def search_patient_by_name(
        # 姓名查询参数
        name: str = Query(..., description="患者姓名（模糊查询）"),

        # 分页参数
        skip: int = Query(0, ge=0),
        limit: int = Query(20, ge=1, le=100),

        db: Session = Depends(get_db)
):
    """按姓名模糊查询患者"""
    query = PatientNameQuery(name=name)
    patients = search_patients_by_name(query, db, skip, limit)

    if not patients:
        raise HTTPException(status_code=404, detail=f"未找到姓名包含'{name}'的患者")

    return patients


@router.get("/", response_model=List[PatientOut])
def get_all_patients_api(
        # 分页参数
        skip: int = Query(0, ge=0, description="跳过前N条数据，默认0"),
        limit: int = Query(20, ge=1, le=100, description="每页显示条数，默认20，最大100"),
        db: Session = Depends(get_db)
):
    """查询所有患者（支持分页）"""
    patients = get_all_patients(db, skip, limit)
    return patients

@router.get("/count", response_model=int)
def get_patients_total_count_api(db: Session = Depends(get_db)):
    """获取所有患者的总数（用于分页计算）"""
    from sqlalchemy import func
    return db.query(func.count(Patient.id)).scalar()