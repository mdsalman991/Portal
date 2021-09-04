pkill -9 -f "redis-server"
pkill -9 -f "uvicorn main:app"
pkill -9 -f "celery -A worker.celery_app worker -l info -Q task-queue -P threads"
