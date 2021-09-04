from worker.celery_app import celery_app
from celery.result import AsyncResult
from fastapi import APIRouter

router = APIRouter()



@router.get("/status/{id}",responses={200:{"description":"Returns the status of the task as (SUCCESS, FAILURE, STARTED, PENDING)","content":{"application/json":{"example":{"status":"FAILURE","success":True}}}}})
async def get_task_status(id:str):
    """
    This endpoint actually gives the status of the task given the task id.
    """
    task = AsyncResult(id,app=celery_app)
    return {"status":task.status,"success":True}

# @router.get("/createmap/{id}",responses={200:{"description":"This is the response for create competency map task","content":{"application/json":{"example":{"id":"0c2dc357-b852-4211-8dcd-9df431a03fcb","success":True}}}}})
# async def create_cmap(id :int):
#     """
#     This is the endpoint for creating competency map
#     """
#     task_name = "worker.celery_app.create_map"
#     task = celery_app.send_task(task_name,args=[id])
#     task_id = task.id
#     return {"id":task_id,"success":True}

@router.get("/createmap2/{id}",responses={200:{"description":"This is the response for create competency map task","content":{"application/json":{"example":{"id":"0c2dc357-b852-4211-8dcd-9df431a03fcb","success":True}}}}})
async def create_cmap2(id:str):
    """
    This is the endpoint for creating competency map
    """
    task_name = "worker.celery_app.create_map2"
    task = celery_app.send_task(task_name,args=[id])
    task_id = task.id
    return {"id":task_id,"success":True}

