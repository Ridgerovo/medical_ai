# 导出当前版本（v1）的所有路由模块
from .doctor import router as doctor_router
from .image import router as image_router  # 新增影像模块路由
from .patient import router as patient_router
from .analysis import router as analysis_router  # 新增分析接口路由
from .report import router as report_router