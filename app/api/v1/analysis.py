from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import subprocess
import json
import os
import sys
import traceback
from datetime import datetime

router = APIRouter()


# ===================== 关键修复：动态检测虚拟环境Python路径 =====================
# 尝试自动检测虚拟环境的Python路径（适应不同环境配置）
def get_virtualenv_python():
    # 尝试1：检查当前激活环境的Python路径（推荐在虚拟环境中启动后端）
    current_python = sys.executable
    if "medical_ai_backend" in current_python:
        return current_python

    # 尝试2：常见的虚拟环境路径（无Scripts目录的情况）
    candidate_paths = [
        r"C:\Users\86157\miniconda3\envs\medical_ai_backend\python.exe",
        r"C:\Users\86157\miniconda3\envs\medical_ai_backend\bin\python.exe",
    ]

    for path in candidate_paths:
        if os.path.exists(path):
            return path

    # 所有尝试失败时，抛出明确错误
    raise FileNotFoundError("未找到medical_ai_backend环境的Python解释器，请检查环境配置")


# 获取并验证Python路径
try:
    VIRTUAL_ENV_PYTHON = get_virtualenv_python()
    print(f"[SUCCESS] 检测到虚拟环境Python路径: {VIRTUAL_ENV_PYTHON}")
except FileNotFoundError as e:
    print(f"[FATAL] 初始化失败: {str(e)}", file=sys.stderr)
    sys.exit(1)


# 定义前端传递的参数模型
class AnalysisRequest(BaseModel):
    image_path: str  # 前端传递的图像路径
    doctor_id: str  # 医生ID
    patient_id: str  # 患者ID


@router.post("/analysis")
def run_analysis(data: AnalysisRequest):
    """触发图像分析程序，处理参数并限制合法路径"""
    try:
        # 1. 定义实际的图片存储目录
        allowed_dir = r"C:\Users\86157\PycharmProjects\medical_ai_backend\data\local_images"
        print(f"[DEBUG] 允许访问的目录: {allowed_dir}")

        # 2. 处理图像路径
        print(f"[DEBUG] 前端传递的image_path: {data.image_path}")
        image_abs_path = os.path.abspath(data.image_path)
        print(f"[DEBUG] image_path的绝对路径: {image_abs_path}")

        # 3. 安全检查：确保图像路径在允许的目录内
        if not image_abs_path.startswith(allowed_dir):
            raise HTTPException(
                status_code=403,
                detail=f"非法的文件路径，仅允许访问: {allowed_dir}，实际路径: {image_abs_path}"
            )

        # 4. 检查图像文件是否存在
        if not os.path.exists(image_abs_path):
            raise HTTPException(
                status_code=400,
                detail=f"图像文件不存在: {image_abs_path}"
            )

        # 5. 调用realtime_predict.py程序
        program_path = os.path.join(
            r"C:\Users\86157\PycharmProjects\medical_ai_backend",
            "Unet",
            "realtime_predict.py"
        )
        print(f"[DEBUG] 分析程序路径: {program_path}")
        if not os.path.exists(program_path):
            raise HTTPException(
                status_code=500,
                detail=f"分析程序不存在: {program_path}"
            )

        # 6. 构造执行命令
        execute_cmd = [
            VIRTUAL_ENV_PYTHON,  # 使用动态检测的Python路径
            program_path,
            image_abs_path
        ]
        print(f"[DEBUG] 实际执行命令: {' '.join(execute_cmd)}")

        # 7. 优化子进程环境（自动适配当前虚拟环境）
        # 获取当前环境的关键变量，确保子进程继承正确的环境
        current_env = os.environ.copy()
        # 确保优先使用当前虚拟环境的依赖
        current_env["PATH"] = f"{os.path.dirname(VIRTUAL_ENV_PYTHON)};{current_env.get('PATH', '')}"

        # 执行子进程
        result = subprocess.run(
            execute_cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=120,
            env=current_env  # 使用当前环境变量，避免隔离导致的问题
        )

        # 8. 检查程序执行是否成功
        if result.returncode != 0:
            print(f"[DEBUG] realtime_predict.py 执行失败！")
            print(f"[DEBUG] 返回码: {result.returncode}")
            print(f"[DEBUG] 程序标准错误输出（stderr）:\n{result.stderr}")
            print(f"[DEBUG] 程序标准输出（stdout）:\n{result.stdout}")
            stderr_detail = result.stderr[:500] if result.stderr else "未知错误（无错误输出）"
            raise HTTPException(
                status_code=500,
                detail=f"分析程序执行失败（返回码: {result.returncode}），错误: {stderr_detail}..."
            )

        # 9. 解析程序输出的JSON结果
        try:
            analysis_result = json.loads(result.stdout)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=500,
                detail=f"解析分析结果失败，程序输出: {result.stdout[:200]}..."
            )

        # 10. 验证分析结果包含必要字段
        required_keys = ["original", "segmented", "comparison"]
        if not all(key in analysis_result for key in required_keys):
            raise HTTPException(
                status_code=500,
                detail=f"分析结果缺少必要字段，实际返回: {list(analysis_result.keys())}"
            )

        # 11. 返回结果给前端
        return {
            "status": "success",
            "received_parameters": {
                "image_path": data.image_path,
                "doctor_id": data.doctor_id,
                "patient_id": data.patient_id
            },
            "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "result_paths": analysis_result
        }

    except HTTPException:
        raise  # 保持原有HTTP异常
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"[FATAL ERROR] 分析接口异常: {str(e)}\n{error_trace}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")
