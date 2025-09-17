import os
import uuid
from pathlib import Path
from sqlalchemy.orm import Session
from fastapi import HTTPException, UploadFile
from app.models.image import MedicalImage
from app.models.doctor import Doctor
from app.models.patient import Patient
from app.schemas.image import ImageCreate
from config import IMAGE_STORAGE  # 从配置文件获取存储目录


def create_medical_image(
        db: Session,
        image_create: ImageCreate,  # 患者/医生ID
        upload_file: UploadFile  # 前端上传的文件流
) -> MedicalImage:
    """
    接收文件流→验证PNG→保存到项目目录→写入数据库
    """
    try:
        # 1. 验证文件是否为PNG（双重校验：后缀+文件头）
        # 校验后缀
        if not upload_file.filename or not upload_file.filename.lower().endswith(".png"):
            raise HTTPException(status_code=400, detail="仅支持PNG格式文件，文件名必须以.png结尾")

        # 校验文件头（避免改后缀的非PNG文件，更严谨）
        try:
            file_header = upload_file.file.read(8)  # 读取前8字节文件头
            upload_file.file.seek(0)  # 读取后重置文件指针（否则后续保存会为空）
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"文件读取失败：{str(e)}")

        if file_header != b'\x89PNG\r\n\x1a\n':  # PNG文件固定头信息
            raise HTTPException(
                status_code=400,
                detail=f"文件不是有效的PNG格式，实际文件头: {file_header.hex()}"
            )

        # 2. 验证医生/患者存在性
        doctor = db.query(Doctor).filter(Doctor.doctor_id == image_create.doctor_id).first()
        if not doctor:
            raise HTTPException(status_code=404, detail=f"医生ID不存在：{image_create.doctor_id}")

        patient = db.query(Patient).filter(Patient.id == image_create.patient_id).first()  # 患者表主键是id
        if not patient:
            raise HTTPException(status_code=404, detail=f"患者ID不存在：{image_create.patient_id}")

        # 3. 后端自动生成：唯一文件名+保存路径（避免覆盖）
        # 生成唯一文件名（原文件名_唯一ID.png）
        original_filename = os.path.splitext(upload_file.filename)[0]
        unique_suffix = uuid.uuid4().hex[:8]
        final_filename = f"{original_filename}_{unique_suffix}.png"

        # 生成保存路径（修复配置引用，使用local_root而非root_dir）
        # 适配config.py中的IMAGE_STORAGE配置
        save_dir = os.path.join(IMAGE_STORAGE["local_root"], unique_suffix)
        try:
            Path(save_dir).mkdir(parents=True, exist_ok=True)  # 自动创建目录
        except PermissionError:
            raise HTTPException(status_code=500, detail=f"没有权限创建目录：{save_dir}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"创建目录失败：{str(e)}")

        final_file_path = os.path.join(save_dir, final_filename)

        # 4. 保存文件到项目目录（写入二进制流）
        try:
            with open(final_file_path, "wb") as f:
                content = upload_file.file.read()
                if not content:
                    raise HTTPException(status_code=400, detail="上传的文件内容为空")
                f.write(content)
        except PermissionError:
            raise HTTPException(status_code=500, detail=f"没有权限写入文件：{final_file_path}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"文件保存失败：{str(e)}")

        # 5. 写入数据库
        image_id = f"IMG-{uuid.uuid4().hex[:12].upper()}"
        new_image = MedicalImage(
            image_id=image_id,
            patient_id=image_create.patient_id,
            patient_name=patient.name,
            file_path=final_file_path,
            file_name=final_filename,
            image_type="PNG",
            doctor_id=image_create.doctor_id,
            doctor_name=doctor.name
        )

        try:
            db.add(new_image)
            db.commit()
            db.refresh(new_image)
        except Exception as e:
            db.rollback()  # 数据库操作失败时回滚
            raise HTTPException(status_code=500, detail=f"数据库写入失败：{str(e)}")

        return new_image

    except HTTPException:
        # 重新抛出已定义的HTTP异常
        raise
    except Exception as e:
        # 捕获所有未处理的异常
        raise HTTPException(status_code=500, detail=f"服务器内部错误：{str(e)}")
