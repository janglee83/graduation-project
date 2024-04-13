import json
from fastapi import APIRouter, HTTPException

router = APIRouter()

@router.get('/')
async def index():
    return { 'data': 'hello' }

