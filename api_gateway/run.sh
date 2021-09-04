mkdir -p logs
redis-server > logs/redis_logs.out &
nohup uvicorn main:app >logs/api_logs.out &
nohup celery -A worker.celery_app worker -l info -Q task-queue -P threads > logs/celery_logs.out &
#celery flower -A worker.celery_app worker -l info -Q task-queue

