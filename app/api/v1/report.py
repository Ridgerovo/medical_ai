from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    UploadFile,
    File,
    Form,
    Path as FastAPIPath,  # 为FastAPI路径参数类取别名，避免与pathlib冲突
    Query
)
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.models.report import Report
from app.schemas.report import ReportResponse, ReportCreateWithFiles,ReportUpdate
import os
import logging
from datetime import datetime
from typing import Optional, List
from pathlib import Path  # 用于文件路径处理的类（保留原名）
from pydantic import BaseModel

# --------------------------
# 基础配置（日志、路由、文件路径）
# --------------------------
# 1. 日志配置（便于排查错误）
logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 2. 初始化路由（仅1次，确保可被导入）
router = APIRouter(
    prefix="/reports",  # 统一路由前缀，避免与其他接口冲突
    tags=["报告管理"]  # 接口文档分类标签
)

# 3. 文件存储路径（绝对路径，跨平台兼容）
# 定位逻辑：report.py → v1 → api → app → 项目根目录 → uploads/reports
PROJECT_ROOT = Path(__file__).parent.parent.parent  # 适配项目目录结构
UPLOAD_DIR = PROJECT_ROOT / "uploads" / "reports"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)  # 自动创建多级目录（不存在则创建）


# --------------------------
# Pydantic模型（数据验证）
# --------------------------

# --------------------------
# 核心接口实现
# --------------------------
@router.post("/upload", response_model=ReportResponse, summary="上传文件创建报告")
def create_report_with_files(
        # 图像文件参数（不变）
        original_image: UploadFile = File(..., description="原始医学影像文件（仅支持PNG/JPG）"),
        analyzed_image: UploadFile = File(..., description="AI分析后影像文件（仅支持PNG/JPG）"),
        # 表单参数（删除parameters，补充report_status，默认0）
        doctor_id: str = Form(..., description="创建报告的医生ID"),
        doctor_name: str = Form(..., description="创建报告的医生姓名"),
        patient_name: str = Form(..., description="报告关联的患者姓名"),
        report_type: str = Form(..., description="报告类型（如：CT/MRI，必填）"),
        description: str = Form(..., description="诊断结论描述（必填）"),
        report_status: int = Form(0, ge=0, le=2, description="报告状态（0-正常，1-审核中，2-已归档，默认0）"),
        # 数据库依赖（不变）
        db: Session = Depends(get_db)
):
    try:
        # 1. 验证文件格式（不变）
        allowed_extensions = ["png", "jpg", "jpeg"]
        original_ext = original_image.filename.split(".")[-1].lower()
        analyzed_ext = analyzed_image.filename.split(".")[-1].lower()
        if original_ext not in allowed_extensions or analyzed_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail="仅支持PNG/JPG/JPEG格式的图像文件"
            )

        # 2. 生成唯一文件名（不变）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 精确到毫秒
        def safe_filename(filename):
            return "".join([c for c in filename if c.isalnum() or c in ('.', '_', '-')])
        original_filename = f"{timestamp}_original_{safe_filename(original_image.filename)}"
        analyzed_filename = f"{timestamp}_analyzed_{safe_filename(analyzed_image.filename)}"

        # 3. 保存文件到服务器（不变）
        original_save_path = UPLOAD_DIR / original_filename
        with open(original_save_path, "wb") as f:
            f.write(original_image.file.read())
        analyzed_save_path = UPLOAD_DIR / analyzed_filename
        with open(analyzed_save_path, "wb") as f:
            f.write(analyzed_image.file.read())

        # 4. 写入数据库（传递report_status，无parameters）
        new_report = Report(
            doctor_id=doctor_id,
            doctor_name=doctor_name,
            patient_name=patient_name,
            report_type=report_type,
            report_status=report_status,  # 补充状态字段（默认0）
            original_image_path=str(original_save_path),
            analyzed_image_path=str(analyzed_save_path),
            description=description,
            create_time=datetime.now()
        )
        db.add(new_report)
        db.commit()
        db.refresh(new_report)

        return new_report

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"创建报告失败：{str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"报告创建失败：{str(e)}"
        )

@router.get("/doctor/{doctor_id}", response_model=List[ReportResponse], summary="按医生ID查询报告")
def get_reports_by_doctor(
        # 路径参数（医生ID）
        doctor_id: str = FastAPIPath(..., description="医生ID"),
        # 分页参数
        skip: int = Query(0, ge=0, description="跳过前N条数据，默认0"),
        limit: int = Query(20, ge=1, le=100, description="每页最大条数，默认20（最大100）"),
        # 数据库依赖
        db: Session = Depends(get_db)
):
    reports = db.query(Report) \
        .filter(Report.doctor_id == doctor_id) \
        .order_by(Report.create_time.desc()) \
        .offset(skip) \
        .limit(limit) \
        .all()
    return reports



@router.patch("/{report_id}", response_model=ReportResponse, summary="更新报告描述和状态")
def update_report(
        # 路径参数（使用FastAPIPath避免命名冲突）
        report_id: str = FastAPIPath(..., description="要更新的报告ID"),
        # 更新数据（通过Pydantic模型验证）
        update_data: ReportUpdate = Depends(),
        # 数据库依赖
        db: Session = Depends(get_db)
):
    # 1. 查询报告是否存在
    db_report = db.query(Report).filter(Report.report_id == report_id).first()
    if not db_report:
        raise HTTPException(
            status_code=404,
            detail=f"报告不存在（ID：{report_id}）"
        )

    try:
        # 2. 更新描述（仅当提供了新描述时）
        if update_data.description is not None:
            db_report.description = update_data.description

        # 3. 更新状态（仅当提供了新状态且状态值合法时）
        if update_data.report_status is not None:
            if update_data.report_status not in [0, 1, 2]:
                raise HTTPException(
                    status_code=400,
                    detail="报告状态仅支持：0（正常）、1（审核中）、2（已归档）"
                )
            db_report.report_status = update_data.report_status

        # 4. 提交更新到数据库
        db.commit()
        db.refresh(db_report)
        return db_report

    except HTTPException as e:
        raise e
    except Exception as e:
        db.rollback()  # 出错时回滚事务
        logger.error(f"更新报告失败（ID：{report_id}）：{str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"报告更新失败：{str(e)}"
        )


@router.delete("/{report_id}", status_code=204, summary="删除报告（含关联文件）")
def delete_report(
        # 路径参数（使用FastAPIPath避免命名冲突）
        report_id: str = FastAPIPath(..., description="要删除的报告ID"),
        # 数据库依赖
        db: Session = Depends(get_db)
):
    # 1. 查询报告是否存在
    db_report = db.query(Report).filter(Report.report_id == report_id).first()
    if not db_report:
        raise HTTPException(
            status_code=404,
            detail=f"报告不存在（ID：{report_id}）"
        )

    try:
        # 2. 删除关联的原始图像文件
        if db_report.original_image_path:
            original_file = Path(db_report.original_image_path)
            # 安全校验：确保删除的是上传目录内的文件（防止越权删除）
            if original_file.is_relative_to(UPLOAD_DIR) and original_file.exists():
                original_file.unlink(missing_ok=True)  # 忽略文件不存在的情况
                logger.info(f"已删除原始图像（报告ID：{report_id}）：{str(original_file)}")
            else:
                logger.warning(f"跳过删除非上传目录文件（报告ID：{report_id}）：{str(original_file)}")

        # 3. 删除关联的分析后图像文件
        if db_report.analyzed_image_path:
            analyzed_file = Path(db_report.analyzed_image_path)
            if analyzed_file.is_relative_to(UPLOAD_DIR) and analyzed_file.exists():
                analyzed_file.unlink(missing_ok=True)
                logger.info(f"已删除分析图像（报告ID：{report_id}）：{str(analyzed_file)}")
            else:
                logger.warning(f"跳过删除非上传目录文件（报告ID：{report_id}）：{str(analyzed_file)}")

        # 4. 删除数据库中的报告记录
        db.delete(db_report)
        db.commit()
        return None  # 204状态码无返回内容

    except HTTPException as e:
        raise e
    except Exception as e:
        db.rollback()
        logger.error(f"删除报告失败（ID：{report_id}）：{str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"报告删除失败：{str(e)}"
        )


@router.get("/", response_model=List[ReportResponse], summary="查询所有报告（分页）")
def get_all_reports(
        # 分页参数（带范围校验）
        skip: int = Query(0, ge=0, description="跳过前N条数据，默认0"),
        limit: int = Query(20, ge=1, le=100, description="每页最大条数，默认20（最大100）"),
        # 数据库依赖
        db: Session = Depends(get_db)
):
    # 按创建时间倒序（最新报告在前）
    reports = db.query(Report) \
        .order_by(Report.create_time.desc()) \
        .offset(skip) \
        .limit(limit) \
        .all()
    return reports


# 在report.py中添加以下代码（建议放在其他接口上方）
@router.get("/{report_id}", response_model=ReportResponse, summary="按报告ID查询详情")
def get_report_by_id(
    report_id: str = FastAPIPath(..., description="报告ID"),
    db: Session = Depends(get_db)
):
    """根据报告ID查询单个报告的详细信息"""
    report = db.query(Report).filter(Report.report_id == report_id).first()
    if not report:
        raise HTTPException(
            status_code=404,
            detail=f"报告不存在（ID：{report_id}）"
        )
    return report


