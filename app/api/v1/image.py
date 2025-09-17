from fastapi import APIRouter, Depends, Query, UploadFile, File, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
import os
from fastapi.responses import FileResponse
import uuid

# 导入项目内部模块（路径需与你的项目结构一致）
from app.db.session import get_db
from app.models.image import MedicalImage  # 你的MedicalImage模型
from app.models.patient import Patient  # 患者模型（用于验证患者存在）
from app.models.doctor import Doctor  # 医生模型（用于验证医生存在）
from app.schemas.image import (  # 影像相关的Pydantic模型
    ImageCreate,
    ImageResponse,
)

from config import IMAGE_STORAGE  # 从配置文件获取影像存储路径

# 结果图片根目录（与realtime_predict.py的RESULTS_FOLDER一致）
RESULTS_ROOT = r"C:\Users\86157\PycharmProjects\medical_ai_backend\Unet\segmentation_results"

# 初始化路由：前缀设为"/medical-images"，最终接口路径为 "/api/v1/medical-images"
router = APIRouter(
    prefix="/medical-images",
    tags=["影像管理"]
)


# --------------------------
# 1. 影像查询接口（核心：前端动态加载medical_image表数据）
# --------------------------
@router.get("/", response_model=List[ImageResponse], summary="获取所有影像数据（支持分页/搜索）")
def get_all_medical_images(
        # 分页参数（前端分页功能依赖）
        skip: int = Query(0, ge=0, description="跳过前N条数据，默认0"),
        limit: int = Query(10, ge=1, le=50, description="每页显示条数，默认10，最大50"),
        # 搜索参数（前端搜索功能依赖）
        search: Optional[str] = Query(None, description="搜索关键词：匹配患者ID/影像ID/文件名"),
        # 筛选参数（前端筛选功能依赖）
        filter_type: Optional[str] = Query("all", description="筛选类型：all=全部，png=仅PNG格式"),
        # 数据库依赖
        db: Session = Depends(get_db)
):
    # 1. 基础查询：按上传时间倒序（最新的在前）
    query = db.query(MedicalImage).order_by(MedicalImage.import_time.desc())

    # 2. 搜索逻辑：匹配患者ID、影像ID、文件名（模糊匹配）
    if search:
        query = query.filter(
            (MedicalImage.patient_id.contains(search)) |
            (MedicalImage.image_id.contains(search)) |
            (MedicalImage.file_name.contains(search))
        )

    # 3. 筛选逻辑：仅PNG格式或全部
    if filter_type == "png":
        query = query.filter(MedicalImage.image_type == "PNG")

    # 4. 分页查询：offset(跳过条数) + limit(显示条数)
    images = query.offset(skip).limit(limit).all()

    # 5. 返回结果（自动通过ImageResponse模型序列化）
    return images


# --------------------------
# 2. 影像上传接口（本地文件上传，前端"本地文件上传"标签页依赖）
# --------------------------
@router.post("/", response_model=ImageResponse, status_code=status.HTTP_201_CREATED,
             summary="上传本地影像（调用add_image逻辑）")
def upload_local_image(
        # 前端表单参数：患者ID、医生ID（必传）
        patient_id: str = File(..., description="关联患者ID（需在t_patient表存在）"),
        doctor_id: str = File(..., description="导入医生ID（需在doctor表存在）"),
        # 前端文件参数：本地PNG文件
        upload_file: UploadFile = File(..., description="PNG格式影像文件", accept="image/png"),
        # 数据库依赖
        db: Session = Depends(get_db)
):
    # 1. 验证文件格式（强制PNG）
    if not upload_file.filename.lower().endswith(".png"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="仅支持PNG格式文件，请重新选择"
        )

    # 2. 验证患者/医生存在性（避免无效ID写入数据库）
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"患者ID不存在：{patient_id}（请确认t_patient表中存在该ID）"
        )

    doctor = db.query(Doctor).filter(Doctor.doctor_id == doctor_id).first()
    if not doctor:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"医生ID不存在：{doctor_id}（请确认doctor表中存在该ID）"
        )

    # 3. 后端自动生成：唯一文件名+存储路径（避免文件覆盖）
    # 生成8位唯一标识（用于目录和文件名去重）
    unique_suffix = uuid.uuid4().hex[:8]
    # 原始文件名（去掉后缀）+ 唯一标识 + .png
    original_filename = os.path.splitext(upload_file.filename)[0]
    final_filename = f"{original_filename}_{unique_suffix}.png"
    # 存储目录（从配置文件读取local_root，如"data/local_images"）
    save_dir = os.path.join(IMAGE_STORAGE["local_root"], unique_suffix)
    os.makedirs(save_dir, exist_ok=True)  # 自动创建目录（不存在则创建）
    final_file_path = os.path.join(save_dir, final_filename)

    # 4. 保存文件到项目目录（写入二进制流）
    try:
        with open(final_file_path, "wb") as f:
            f.write(upload_file.file.read())
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"文件保存失败：{str(e)}"
        )

    # 5. 写入medical_image表（字段与模型完全对齐）
    image_id = f"IMG-{uuid.uuid4().hex[:12].upper()}"  # 生成影像唯一ID
    new_image = MedicalImage(
        image_id=image_id,
        patient_id=patient_id,
        patient_name=patient.name,  # 从患者表获取真实姓名
        file_path=final_file_path,  # 后端生成的存储路径
        file_name=final_filename,  # 后端生成的唯一文件名
        image_type="PNG",  # 强制PNG格式
        doctor_id=doctor_id,
        doctor_name=doctor.name,  # 从医生表获取真实姓名
        import_time=datetime.now()  # 上传时间（也可依赖数据库默认值）
    )

    db.add(new_image)
    db.commit()
    db.refresh(new_image)  # 刷新数据，获取数据库自动生成的字段（如import_time）

    return new_image


# --------------------------
# 3. 查询所有影像记录（无分页，返回全部数据）
# --------------------------
@router.get("/all", response_model=List[ImageResponse], summary="获取所有影像记录（无分页）")
def get_all_images(
        db: Session = Depends(get_db)
):
    """查询medical_image表中所有影像记录（不做分页处理）"""
    all_images = db.query(MedicalImage).order_by(MedicalImage.import_time.desc()).all()
    return all_images


# --------------------------
# 4. 删除指定影像记录
# --------------------------
@router.delete("/{image_id}", status_code=status.HTTP_204_NO_CONTENT, summary="删除指定ID的影像记录")
def delete_image(
        image_id: str,  # 路径参数：影像唯一ID
        db: Session = Depends(get_db)
):
    """根据image_id删除影像记录，同时删除本地文件"""
    # 1. 查询影像是否存在
    image = db.query(MedicalImage).filter(MedicalImage.image_id == image_id).first()
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"影像ID不存在：{image_id}"
        )

    # 2. 删除本地文件（可选，根据业务需求决定是否保留文件）
    if os.path.exists(image.file_path):
        try:
            os.remove(image.file_path)
            # 如果文件所在目录为空，可一并删除目录
            dir_path = os.path.dirname(image.file_path)
            if not os.listdir(dir_path):
                os.rmdir(dir_path)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"删除本地文件失败：{str(e)}"
            )

    # 3. 删除数据库记录
    db.delete(image)
    db.commit()

    # 4. 204状态码无返回内容
    return None


# --------------------------
# 5. 通过文件路径获取图片
# --------------------------
@router.get("/file", summary="通过文件路径获取图片")
def get_image_by_file_path(
        file_path: str = Query(..., description="图片在服务器的本地路径（如data/local_images/xxx/xxx.png）"),
        db: Session = Depends(get_db)
):
    """
    前端通过此接口获取图片：
    1. 验证文件路径是否存在
    2. 验证路径是否属于系统配置的存储目录（安全校验）
    3. 返回图片文件流
    """
    # 1. 安全校验：确保访问的文件在配置的存储目录内（防止目录遍历攻击）

    from app.api.v1.report import UPLOAD_DIR

    allowed_roots = [
        IMAGE_STORAGE["local_root"],  # 本地上传的图片根目录
        IMAGE_STORAGE["pacs_root"],  # PACS导入的图片根目录
        str(UPLOAD_DIR)
    ]
    # 将路径转换为绝对路径，统一处理分隔符
    file_path = os.path.abspath(file_path)
    # 检查是否在允许的目录内
    if not any(file_path.startswith(os.path.abspath(root)) for root in allowed_roots):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="访问的文件路径不允许"
        )

    # 2. 验证文件是否存在
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"图片文件不存在：{file_path}"
        )

    # 3. 验证是否为图片文件（防止返回非图片文件）
    if not file_path.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="仅支持PNG/JPG/JPEG格式的图片文件"
        )

    # 4. 返回图片文件流（自动处理MIME类型）
    return FileResponse(
        path=file_path,
        filename=os.path.basename(file_path),  # 浏览器下载时的默认文件名
        media_type="image/png"  # 固定为PNG（因项目只支持PNG）
    )


# --------------------------
# 6. AI分析结果图片接口（关键：移到/{image_id}接口上方，避免路由冲突）
# --------------------------
@router.get("/analysis-result-image")
def get_analysis_result_image(file_name: str):
    # 1. 安全校验：防止路径遍历攻击
    results_root_abs = os.path.abspath(RESULTS_ROOT)
    image_full_path = os.path.abspath(os.path.join(RESULTS_ROOT, file_name))

    # 确保拼接后的路径仍在结果目录内
    if not image_full_path.startswith(results_root_abs):
        raise HTTPException(
            status_code=403,
            detail="文件名包含非法路径，拒绝访问"
        )

    # 2. 调试日志：确认路径和文件列表
    print(f"[DEBUG] 结果目录绝对路径: {results_root_abs}")
    print(f"[DEBUG] 图片实际查找路径: {image_full_path}")
    try:
        dir_files = os.listdir(results_root_abs)
        print(f"[DEBUG] 结果目录文件列表: {dir_files}")
    except Exception as e:
        print(f"[DEBUG] 读取目录失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"无法访问结果目录: {str(e)}"
        )

    # 3. 验证文件存在性
    if not os.path.exists(image_full_path):
        raise HTTPException(
            status_code=404,
            detail=f"图片不存在！实际查找路径：{image_full_path}"
        )

    # 4. 验证文件格式
    if not image_full_path.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(
            status_code=400,
            detail="仅支持PNG/JPG/JPEG格式的图片文件"
        )

    # 5. 返回图片流
    return FileResponse(
        path=image_full_path,
        filename=os.path.basename(image_full_path),
        media_type="image/png"
    )


# --------------------------
# 7. 根据影像ID查询详情（关键：放在analysis-result-image接口下方，避免冲突）
# --------------------------
@router.get("/{image_id}", response_model=ImageResponse, summary="根据影像ID查询详情")
def get_image_by_id(
        image_id: str,  # 路径参数：影像唯一ID
        db: Session = Depends(get_db)
):
    """根据影像ID查询详细信息，用于影像详情页展示"""
    # 查询影像是否存在
    image = db.query(MedicalImage).filter(MedicalImage.image_id == image_id).first()
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"影像IDD不存在：{image_id}"
        )
    return image