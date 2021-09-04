const express = require("express");
const router = express.Router();

// MIDDLEWARE
const Authenticator = require("../middleware/authenticator");

// MODELS
const Task = require("../model/tasks");
const ObjectId = require("mongodb").ObjectID;
const LearningMap = require("../model/learningMap");


// HTTP
const http = require("http");

// ROUTES

router.get('/getMapStatus', Authenticator.ensureAuthenticated, async (req, res) => {

  const objectId = req.query.objectId;

  const task = await Task.findTaskByObjectId({
    "id": ObjectId(objectId)
  });
  
  // console.log(`TASK ID:  ${task.taskId}`);
  
  if (task != null) {    

    const options = {
      host: "localhost",
      port: 8000,
      path: `/cmap/status/${task.taskId}`
    };

    http.get(options, (response) => {
      // console.log(`STATUS CODE: ${response.statusCode}`);
      let body = '';
      let status;
      response.on('data', function (chunk) {
        body += chunk;
      });
      response.on('end', async function () {
        body = JSON.parse(body);
        status = body["status"];
        
        // console.log(`STATUS: ${status}`);

        // Update status if it has changed
        if (task.taskStatus.localeCompare(status)) {
          await Task.updateTaskByTaskId({
            "taskId": task.taskId,
            "taskStatus": status
          });
        }
        res.send(status);
      });
    });
  } else {
    res.send("UNKNOWN");
  }
});

module.exports = router;