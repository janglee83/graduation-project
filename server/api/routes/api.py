from fastapi import APIRouter

from api.routes import predictor
from api.routes import core

router = APIRouter()
router.include_router(predictor.router, tags=["predictor"], prefix="/v1")
router.include_router(core.router, tags=["core"], prefix='/v1')
