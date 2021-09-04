// Node packages
const { parseAsync } = require("json2csv");
const fs = require("fs");
const mongoose = require("mongoose");

// MongoDB models
const User = require("../model/user");
const Logs = require("../model/log");
const LearningMap = require("../model/learningMap");
const LearningMapResources = require("../model/learningMapResources");

// Database configuration & connection
const appConfig = require("../config/app_config");
const Log = require("../model/log");
const mongoDBConnectionURL = `mongodb://${appConfig.MONGO_HOST}/${appConfig.MONGO_DB}`;
mongoose.connect(mongoDBConnectionURL);

// BELOW CODE IS INEFFICIENT FOR LARGE DATASETS -- NEEDS WORK!

function getReadableDateAndTime(isoDate) {
  date = new Date(isoDate);
  year = date.getFullYear();
  month = date.getMonth() + 1;
  dt = date.getDate();
  hours = date.getHours();
  minutes = date.getMinutes();
  seconds = date.getSeconds();

  if (dt < 10) {
    dt = "0" + dt;
  }
  if (month < 10) {
    month = "0" + month;
  }
  if (hours < 10) {
    hours = "0" + hours;
  }
  if (minutes < 10) {
    minutes = "0" + minutes;
  }
  if (seconds < 10) {
    seconds = "0" + seconds;
  }
  return {
    date: dt + "-" + month + "-" + year,
    time: hours + ":" + minutes + ":" + seconds,
  };
}

async function resolve() {
  let allLogs = await Logs.find().lean();
  const allUsers = await User.find().lean();
  const allLearningMaps = await LearningMap.find().lean();
  const allLearningMapResources = await LearningMapResources.find().lean();

  // Resolve user in logs and convert time to a readable format
  allLogs = allLogs.map((log) => {
    for (let iter = 0; iter < allUsers.length; iter++) {
      if (log["userId"].toString() === allUsers[iter]["_id"].toString()) {
        const { date, time } = getReadableDateAndTime(log.logTime);

        delete log.__v;
        delete log._id;
        delete log.logTime;
        return {
          ...log,
          date,
          time,
          username: allUsers[iter]["username"],
        };
      }
    }
  });

  // Resolve map name in logs
  allLogs = allLogs.map((log) => {
    for (let iter = 0; iter < allLearningMaps.length; iter++) {
      if (
        log["learningMapId"].toString() ===
        allLearningMaps[iter]["_id"].toString()
      ) {
        return {
          ...log,
          mapName: allLearningMaps[iter]["mapName"],
        };
      }
    }
  });

  // Resolve all map resources in logs
  allLogs = allLogs.map((log) => {
    for (let iter = 0; iter < allLearningMapResources.length; iter++) {
      if (
        log["learningMapResourceId"].toString() ===
        allLearningMapResources[iter]["_id"].toString()
      ) {
        return {
          ...log,
          resourceName: allLearningMapResources[iter]["resourceId"],
          locationX: allLearningMapResources[iter]["locationX"],
          locationY: allLearningMapResources[iter]["locationY"],
          normX: allLearningMapResources[iter]["normX"],
          normY: allLearningMapResources[iter]["normY"],
        };
      }
    }
  });

  console.log(allLogs);

  // Transform data
  allLogs.forEach((log) => {
    log.userId = log.userId.toString();
    log.learningMapId = log.learningMapId.toString();
    log.learningMapResourceId = log.learningMapResourceId.toString();
  });

  /*
  result.forEach(record => {
  record.zip = parseFloat(record.zip);
  record.sq__ft = parseFloat(record.sq__ft);
  record.baths = parseFloat(record.baths);
  record.beds = parseFloat(record.beds);
  record.price = parseFloat(record.price);
  record.latitude = parseFloat(record.latitude);
  record.longitude = parseFloat(record.longitude);
});
  */

  // Define fields in CSV
  const fields = [
    "learningMapId",
    "mapName",
    "learningMapResourceId",
    "resourceName",
    "locationX",
    "locationY",
    "normX",
    "normY",
    "date",
    "time",
    "userId",
    "username",
  ];
  const resultCSV = await parseAsync(allLogs, { fields });
  fs.writeFileSync("./log_file.csv", resultCSV);
}

resolve().then(() => {
  process.exit(1);
});
