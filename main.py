import uvicorn
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from config import API_PREFIX
from app.api.v1.doctor import router as doctor_router
from app.api.v1.image import router as image_router
from app.api.v1.patient import router as patient_router
from app.api.v1.analysis import router as analysis_router
from app.api.v1.report import router as report_router
# 假设需要登录校验（示例，需与实际登录逻辑匹配）
from app.services.auth import get_current_doctor

# 创建FastAPI应用
app = FastAPI(
    title="医疗AI后端系统",
    description="支持医生管理、影像导入的医疗AI服务",
    version="1.0.0"
)

# 1. 挂载静态文件（前端页面、CSS、JS等）
app.mount("/static", StaticFiles(directory="static"), name="static")


# 3. 注册后端API路由
app.include_router(doctor_router, prefix=API_PREFIX)
app.include_router(image_router, prefix=API_PREFIX)
app.include_router(patient_router, prefix=API_PREFIX)
app.include_router(analysis_router, prefix=API_PREFIX)
app.include_router(report_router,prefix=API_PREFIX)
# 4. 根路径重定向到前端登录页
@app.get("/", response_class=RedirectResponse)
async def read_root():
    return RedirectResponse(url="/static/login.html")  # 前端登录页路径


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)