from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

# 导入项目内部模块
from app.db.session import get_db
from app.models.image import MedicalImage
from app.models.patient import Patient
from app.models.report import Report

# 初始化路由
router = APIRouter(
    prefix="/stats",
    tags=["数据统计"]
)


@router.get("/total-images", response_model=int, summary="统计总影像数")
def get_total_images(db: Session = Depends(get_db)):
    """获取系统中所有医学影像的总数"""
    return db.query(func.count(MedicalImage.image_id)).scalar() or 0


@router.get("/total-patients", response_model=int, summary="统计总患者数")
def get_total_patients(db: Session = Depends(get_db)):
    """获取系统中所有患者的总数"""
    return db.query(func.count(Patient.id)).scalar() or 0


@router.get("/total-reports", response_model=int, summary="统计总报告数")
def get_total_reports(db: Session = Depends(get_db)):
    """获取系统中所有诊断报告的总数"""
    return db.query(func.count(Report.report_id)).scalar() or 0


@router.get("/pending-tasks", response_model=int, summary="统计待处理任务数")
def get_pending_tasks(db: Session = Depends(get_db)):
    """获取状态为待审核的报告数量（待处理任务）"""
    return db.query(func.count(Report.report_id)) \
        .filter(Report.report_status == 1) \
        .scalar() or 0


@router.get("/dashboard", summary="获取仪表盘所有统计数据")
def get_dashboard_stats(db: Session = Depends(get_db)):
    """一次性获取仪表盘所需的所有统计数据"""
    return {
        "total_images": db.query(func.count(MedicalImage.image_id)).scalar() or 0,
        "total_patients": db.query(func.count(Patient.id)).scalar() or 0,
        "total_reports": db.query(func.count(Report.report_id)).scalar() or 0,
        "pending_tasks": db.query(func.count(Report.report_id))
            .filter(Report.report_status == 1)
            .scalar() or 0
    }