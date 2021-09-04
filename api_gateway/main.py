import os
import logging
from fastapi import FastAPI, BackgroundTasks


from routers import cmap, system
from Discussion_router import discussion


app = FastAPI(docs_url="/")

app.include_router(cmap.router,prefix="/cmap",tags=['Competency Maps'])
app.include_router(system.router,prefix="/sys",tags=['Server and System'])
app.include_router(discussion.router,prefix="/disc",tags=['Discussion Analyzer'])


# @app.get("/")
# async def root():
    # return "Home screen"
