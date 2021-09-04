from fastapi import status, APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()

@router.get("/health")
async def health():
    "This API actually gives a response incase the server is up and running"
    # return JSONResponse(status_code=status.HTTP_200_OK)
    return {'status':'success','serverup':True}
