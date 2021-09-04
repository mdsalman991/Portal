const express = require("express");
const session = require("express-session");
const router = express.Router();

const Log = require("../model/log");
const User = require("../model/user");
const Authenticator = require("../middleware/authenticator");

// Records a user's click on a resource (on the map)
router.get(
  "/logResourceClick",
  Authenticator.ensureAuthenticated,
  (req, res) => {
    const resource = req.query.resource;

    const newLog = new Log({
      userId: req.user._id,
      learningMapId: resource.mapId,
      learningMapResourceId: resource._id,
    });

    Log.createLog(newLog, (error, log) => {
      if (error) throw error;
    });
  }
);

// Fetches all the logs for a specific list of resources
router.get(
  "/getResourcesLog",
  Authenticator.ensureAuthenticated,
  (req, res) => {
    Log.findResourceLog({
        resourceIds: req.query.resourceIds
      },
      (error, logs) => {
        if (error) throw error;
        res.send(logs);
      }
    );
  }
);

router.get(
  "/getAllUsersRecentLog",
  Authenticator.ensureAuthenticated,
  async (req, res) => {
    try {
      const users = await User.getAllUsers();

      const allLogs = [];

      for (let iter = 0; iter < users.length; iter++) {
        try {
          const logs = await Log.findLastNLogForUser(users[iter]._id, 3);
          allLogs.push({
            userId: users[iter]._id.toString(),
            username: users[iter].username.toString(),
            logs: logs,
            isCurrentUser: users[iter]._id.toString() === req.user._id.toString() ?
              true : false,
          });
        } catch (exception) {}
      }

      res.send(allLogs);
    } catch (exception) {
      console.log(exception);
      res.send("Error fetching users!");
    }
  }
);

module.exports = router;