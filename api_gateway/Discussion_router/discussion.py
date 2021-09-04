from worker.celery_app import celery_app
from celery.result import AsyncResult
from fastapi import APIRouter
from Discussion_router import fun
from typing import Dict, Any

router = APIRouter()
# API called to check if intervener is needed for the discussion forum
# y is 0 or 1, 1 denotes intervention is needed
@router.post("/result")
async def result(item : Dict[Any, Any]):
    # index is used to denote which answer thread of the question is being queried 
    y,bi = fun.getResult(item['discussion'],item['index']) 
    return y

# bi is 0 or 1 or 2, which indicate Disagreement,partial,agreement respectively 
@router.post("/branch")
async def branch(item : Dict[Any, Any]):
    # index is used to denote which answer thread of the question is being queried 
    y,bi = fun.getResult(item['discussion'],item['index']) 
    return bi
    
# if __name__ == "__main__":
#     uvicorn.run(app,host = '127.0.0.1',port=8080)