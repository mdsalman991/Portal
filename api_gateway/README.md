README

#SETUP

Please run the setup.sh file (assuming you have a new conda environment). this will install all the dependencies.
> setup.sh

Check if redis is installed incase its not please follow the procedure here https://redis.io/topics/quickstart.

To check if redis is running
> redis-cli ping 
This will return 
> 'PONG'
If its not running then use the following command 
> redis-server

After everything is setup. from the current folder (api_gateway/) open two terminal sessions. 
one of them will run the fast api app and another will run the celery. 

Session - I
> uvicorn main:app
Session - II
> celery -A worker.celery_app worker -l info -Q task-queue -P threads


**ALTERNATIVELY**,
 you can also run the 
> ./run.sh

So that the worker and FAST API run in the background. The logs are put in the logs/ directory.

you can open the following link for checking the Swagger docs.

https://localhost:8000/docs

Detailed Documentation is in this documentation 
https://docs.google.com/document/d/1mAqtI3F6J13bA0TLuw-rAViFcyP482pVybyPnj7rkl0/edit?usp=sharing

------------------------------------------------------------------------------------------
##OPTIONAL : 
We can use flower to actually track the tasks using GUI. For flower you have to install the package using pip and then run as follows 

> celery flower -A worker.celery_app worker -l info -Q task-queue 

For running flower there can be various commands. Hence don't get confused if there is another similar command that works for runnign this.

NOTE : Flower must be started only after the celery worker is setup.
---------------------------------------------------------------------------------------------
##TERMINATION : 
For stopping the applications please kill each of them using kill -9 command and theire respective PIDs.

> kill -9 {all PIDS}

For removing unnecessary bytecodes run 
> ./clean.sh
Shutdown the redis server 
> redis-cli shutdown

