var express = require("express");
var router = express.Router();
var ObjectId = require("mongodb").ObjectID;
var User = require("../model/user");
const multer = require("multer");
var path = require("path");
var Resource = require("../model/resource");
var LearningMap = require("../model/learningMap");
var LearningMapResources = require("../model/learningMapResources");
var LearningMapUsers = require("../model/learningMapUsers");
var CmapResourceTopics = require("../model/cmapResourceTopics");
var LearningPathways = require("../model/learningPathways");
var Authenticator = require("../middleware/authenticator");
const CompetencyMap = require("../model/competencyMap");
const CmapResources = require("../model/cmapResources");
const CmapTopics = require("../model/cmapTopics");
var ResourceGroup = require("../model/resourcegroup");
var Resource = require("../model/resource");
const http = require("http");
const Task = require("../model/tasks");
const config = require("../config/app_config");
if (typeof localStorage === "undefined" || localStorage === null) {
  var LocalStorage = require('node-localstorage').LocalStorage;
  localStorage = new LocalStorage('./scratch');
}

var multerStorage = multer.diskStorage({
  destination: function (req, file, callback) {
    callback(null, path.join(__dirname, "../public/trailer_generation/template_2/data/default/author/"));
  },
  filename: function (req, file, callback) {
    callback(null,file.originalname);
  },
});

router.get(
  "/viewAllMaps",
  Authenticator.ensureAuthenticated,
  async (req, res) => {
    var learningMapsList = await LearningMap.getAllLearningMaps();
    var modifiedList = [];
    for (var i = 0; i < learningMapsList.length; i++) {
      var learningMap = learningMapsList[i];
      var learningMapId = learningMap._id;
      var LearningMapResourcesList = await LearningMapResources.getAllLearningMapResources(
        learningMapId
      );
      learningMap.noOfResources = LearningMapResourcesList.length;

      var totalAttentionTime = 0;
      var resourceTypesobject = {
        Text: 0,
        Pdf: 0,
        Video: 0,
      };

      for (var j = 0; j < LearningMapResourcesList.length; j++) {
        var LearningMapResource = LearningMapResourcesList[j];
        // TODO : Change to resourceName
        const resourceId = LearningMapResource.resourceId;
        const resource = await Resource.getResourceByResourceId(resourceId);
        if (resource) {
          totalAttentionTime += resource.attentionTime;
          resourceTypesobject[resource.resourceType] = 1;
        }
      }

      learningMap.totalAttentionTime = totalAttentionTime;
      learningMap.resourceTypesobject = resourceTypesobject;

      var userList = await LearningMapUsers.getLearningMapUsers(
        ObjectId(learningMapId)
      );
      learningMap.userList = userList;

      var createdById = learningMap.createdBy;
      // BELOW LINE MAY NOT WORK IF USER FOR GIVEN _ID DOESN'T EXIST
      var createdByUsername = await User.getUserByUserId(ObjectId(createdById));
      learningMap.createdByUsername = createdByUsername[0].username;

      var LearningPathwaysList = await LearningPathways.getLearningPathwaysByMapId(
        learningMapId
      );
      var unique = [];
      var distinct = [];
      for (let i = 0; i < LearningPathwaysList.length; i++) {
        if (!unique[LearningPathwaysList[i].pathwayId]) {
          distinct.push(LearningPathwaysList[i].pathwayId);
          unique[LearningPathwaysList[i].pathwayId] = 1;
        }
      }

      learningMap.noOfPathways = distinct.length;
      modifiedList.push(learningMap);
    }

    res.render("viewAllMaps", {
      layout: "layoutViewAllMaps",
      user: req.user,
      username: req.user.username,
      learningMapsList: modifiedList,
    });
  }
);

router.get(
  "/updateSubscribers",
  Authenticator.ensureAuthenticated,
  async (req, res) => {
    var learningMapId = req.query.learningMapId;
    var userId = req.query.userId;

    var newLearningMapUser = new LearningMapUsers({
      mapId: ObjectId(learningMapId),
      userId: ObjectId(userId),
    });

    var createLearningMapUser = await LearningMapUsers.createLearningMapUsers(
      newLearningMapUser
    );
    var userList = await LearningMapUsers.getLearningMapUsers(
      ObjectId(learningMapId)
    );
    var noOfSubscribers = userList.length;

    res.send({
      subscribersCount: noOfSubscribers
    });
  }
);

router.get("/viewMap", Authenticator.ensureAuthenticated, async (req, res) => {
  var learningMapId = req.query.learningMapId;
  var map = await LearningMap.getCMapId(ObjectId(learningMapId));
  if (map == null || map == undefined || map.length === 0) {
    res.redirect("/learningMaps/viewAllMaps");
  } else {
    var cmapId = map[0].cMapId;
    var learningMapResources = await LearningMapResources.getAllLearningMapResources(
      ObjectId(learningMapId)
    );
    var paths = await LearningPathways.getAllPaths(ObjectId(learningMapId));
    var topics = JSON.stringify(
      await CmapResourceTopics.getAllTopics(ObjectId(cmapId))
    );
    var resMap = {};
    for (var i = 0; i < learningMapResources.length; i++) {
      val = JSON.parse(JSON.stringify(learningMapResources[i]));
      val["pathwayId"] = "";
      val["sequenceId"] = 0;
      val["pathwayLod"] = 1;
      const resource = await Resource.getResourceByResourceId(val.resourceId);
      val["resourceName"] = resource.resourceName;
      resMap[val.resourceId] = val;
    }
    var path_data = [],
      res_data = [];
    var pathMap = {};
    for (var i = 0; i < paths.length; i++) {
      val = resMap[paths[i].resourceId];
      if (val !== undefined) {
        val["pathwayId"] = paths[i].pathwayId;
        val["sequenceId"] = paths[i].sequenceId;
        val["pathwayLod"] = paths[i].pathwayLod;

        if (paths[i].pathwayId in pathMap) pathMap[paths[i].pathwayId].push(val);
        else pathMap[paths[i].pathwayId] = [val];
        res_data.push(val);
        delete resMap[paths[i].resourceId];
      }
    }
    for (i in pathMap) {
      path_data.push({
        values: pathMap[i]
      });
    }
    for (i in resMap) {
      res_data.push(resMap[i]);
    }

    Resource.getFileType({
        resourceIds: res_data.map((d) => d.resourceId)
      },
      (error, types) => {
        if (error) throw error;

        for (let iter = 0; iter < res_data.length; iter++) {
          let result = types.filter(
            // ! Changed to use resourceName instead of resourceId
            // (d) => d.resourceName === res_data[iter]["resourceId"]
            (d) => d.resourceName === res_data[iter]["resourceName"]
          )[0];
          let resourceType;
          if (result !== undefined) {
            resourceType = result._doc.resourceType; // code is weird!
          } else {
            // Just writing a wierd workaround
            // ! Changed to use resourceName instead of resourceId
            /* resourceType = res_data[iter]["resourceId"].includes(".pdf") ?
              "Pdf" :
              res_data[iter]["resourceId"].includes(".txt") ?
              "Text" :
              "Video";
              */

            resourceType = res_data[iter]["resourceName"].includes(".pdf") ?
              "Pdf" :
              res_data[iter]["resourceName"].includes(".txt") ?
              "Text" :
              "Video";
          }

          res_data[iter]["resourceType"] = resourceType;
        }
        localStorage.setItem("mapId",JSON.stringify(learningMapId));
        res.render("viewMap", {
          layout: "layoutViewMap",
          mapId:JSON.stringify(learningMapId),
          name: map[0].mapName,
          description: map[0].mapDescription,
          username: req.user.username,
          pathData: JSON.stringify(path_data),
          resData: JSON.stringify(res_data),
          topic: topics,
        });
      }
    );
  }

});

router.get("/newMap", Authenticator.ensureAuthenticated, async (req, res) => {
  res.render("newMap", {
    layout: "layoutNewMap",
    username: req.user.username
  });
});

router.get(
  "/viewMyMaps",
  Authenticator.ensureAuthenticated,
  async (req, res) => {
    var userId = req.user._id;
    var learningMapsUserList = await LearningMapUsers.getLearningMapsByUser(
      userId
    );
    var modifiedList = [];
    for (var i = 0; i < learningMapsUserList.length; i++) {
      var learningMapUserId = learningMapsUserList[i].mapId;
      var learningMapList = await LearningMap.getLearningMapById(
        ObjectId(learningMapUserId)
      );
      var learningMap = learningMapList[0];
      var learningMapId = learningMap._id;
      var LearningMapResourcesList = await LearningMapResources.getAllLearningMapResources(
        learningMapId
      );
      learningMap.noOfResources = LearningMapResourcesList.length;

      var totalAttentionTime = 0;
      var resourceTypesobject = {
        Text: 0,
        Pdf: 0,
        Video: 0,
      };
      for (var j = 0; j < LearningMapResourcesList.length; j++) {
        var LearningMapResource = LearningMapResourcesList[j];
        var rsrname = LearningMapResource.resourceId;
        rsrname = rsrname.split(".")[0];
        var resource = await Resource.getResourceByResourceNameRegex(rsrname);
        if (resource) {
          totalAttentionTime += resource.attentionTime;
          resourceTypesobject[resource.resourceType] = 1;
        }
      }

      learningMap.totalAttentionTime = totalAttentionTime;
      learningMap.resourceTypesobject = resourceTypesobject;

      var userList = await LearningMapUsers.getLearningMapUsers(
        ObjectId(learningMapId)
      );
      learningMap.userList = userList;

      var createdById = learningMap.createdBy;
      var createdByUsername = await User.getUserByUserId(ObjectId(createdById));
      learningMap.createdByUsername = createdByUsername[0].username;

      var LearningPathwaysList = await LearningPathways.getLearningPathwaysByMapId(
        ObjectId(learningMapId)
      );
      var unique = [];
      var distinct = [];
      for (let i = 0; i < LearningPathwaysList.length; i++) {
        if (!unique[LearningPathwaysList[i].pathwayId]) {
          distinct.push(LearningPathwaysList[i].pathwayId);
          unique[LearningPathwaysList[i].pathwayId] = 1;
        }
      }
      learningMap.noOfPathways = distinct.length;

      modifiedList.push(learningMap);
    }

    res.render("viewMyMaps", {
      layout: "layoutViewAllMaps",
      user: req.user,
      username: req.user.username,
      learningMapsList: modifiedList,
    });
  }
);

// RESOURCE SELECTION
router.get(
  "/createMap",
  Authenticator.ensureAuthenticated,
  async (req, res) => {
    var resourceGroupList = await ResourceGroup.getAllResourceGroup();

    for (let iter = 0; iter < resourceGroupList.length; iter++) {
      let resources = await Resource.getResourceListByResourceId(
        resourceGroupList[iter]["resources"].map(
          (resource) => resource["resourceId"]
        )
      );
      // Filter the contents of the resources
      resources = resources.map((resource) => {
        let filteredResource = {
          attentionTime: resource.attentionTime,
          resourceType: resource.resourceType,
          resourceName: resource.resourceName,
          resourceId: resource._id,
        };
        return filteredResource;
      });

      resourceGroupList[iter]["resources"] = resources;
    }

    // console.log(resourceGroupList[0]);
    res.render("createMap", {
      layout: "layoutResourceSelection",
      resourceGroupList: resourceGroupList,
      username: req.user.username,
    });
  }
);

router.post(
  "/createMap",
  Authenticator.ensureAuthenticated,
  async (req, res) => {
    //console.log(req.body);
    const mapName = req.body.mapName;
    const mapDescription = req.body.mapDescription;
    const numberOfLevels = +req.body.numberOfLevels;
    let resourceIDList;
    if (Array.isArray(req.body.resources))
      resourceIDList = req.body.resources;
    else
      resourceIDList = [req.body.resources];

    const user = req.user;

    const competencyMap = new CompetencyMap({
      mapName: mapName,
      numLevels: numberOfLevels,
      numResources: resourceIDList.length,
      createdBy: ObjectId(user._id),
    });

    CompetencyMap.createCompetencyMap(competencyMap).then((cmap) => {
      resourceIDList.forEach(async (resourceID) => {
        const cmapResources = new CmapResources({
          mapId: ObjectId(cmap._id),
          resourceId: resourceID,
        });

        await CmapResources.createCmapResources(cmapResources);
      });

      const learningMap = new LearningMap({
        mapName: mapName,
        mapDescription: mapDescription,
        cMapId: ObjectId(cmap._id),
        createdBy: ObjectId(user._id),
      });

      LearningMap.createLearningMap(learningMap).then((lmap) => {
        // CALL API END POINT
        const options = {
          host: "localhost",
          port: 8000,
          path: `/cmap/createmap2/${cmap._id}`,
        };
        http.get(options, (response) => {
          console.log(`STATUS CODE: ${response.statusCode}`);
          let body = '';
          let taskID;
          response.on('data', function (chunk) {
            body += chunk;
          });
          response.on('end', function () {
            body = JSON.parse(body);
            taskID = body["id"];
            console.log(`TASK ID: ${taskID}`);

            const newTask = new Task({
              taskType: 'create_map',
              taskId: taskID,
              taskStatus: 'STARTED',
              objectId: lmap._id,
            });

            Task.createTask(newTask, (error, task) => {
              if (error) throw error;
            });
          });
        });
        res.redirect("/learningMaps/viewAllMaps");
      });
    });
  }
);

router.post("/deleteMap", Authenticator.ensureAuthenticated, async (req, res) => {
  const learningMapId = req.body.learningMapId;
  const learningMapObject = await LearningMap.getLearningMapById(learningMapId);
  if (learningMapObject) {
    const invokingUser = req.user._id;
    const learningMapCreator = learningMapObject[0]["createdBy"];
    if (invokingUser.toString() === learningMapCreator.toString()) {
      // Logged in user is the owner of the map
      console.log("Intiating Deletion!");
      const competencyMapId = learningMapObject[0].cMapId;
      await LearningMap.deleteLearningMapById(learningMapId);
      await LearningMapResources.deleteLearningMapResourcesByMapId(learningMapId);
      await LearningMapUsers.deleteLearningMapUsersByMapId(learningMapId);
      await LearningPathways.deleteLearningPathwaysByMapId(learningMapId);
      await CompetencyMap.deleteCompetencyCMapById(competencyMapId);
      await CmapResources.deleteCmapResourcesByCMapId(competencyMapId);
      await CmapResourceTopics.deleteCmapResourceTopicsByCMapId(competencyMapId);
      await CmapTopics.deleteCmapTopicsByCMapId(competencyMapId);
      await Task.deleteTaskByObjectId(learningMapId);
      // Delete folder from `api_gateway/output/`
      require('child_process').exec(`rm -rf ${config.API_GATEWAY_OUTPUT}/${competencyMapId}`)
      console.log("Deletion Complete!!");
    }
  }
  res.redirect("/learningMaps/viewAllMaps");
});

router.get(
  "/trailerGeneration",
  Authenticator.ensureAuthenticated,
  async (req, res) => {
    var learningMapsLst = await LearningMap.getAllLearningMaps();
    res.render("trailerGeneration", {
      layout: "layoutTrailerGeneration",
      user: req.user,
      username: req.user.username,
      learningMapsList: learningMapsLst,
    }); 
  }
);
router.get("/fetchPaths",Authenticator.ensureAuthenticated, async (req, res) => {
  const mapId = req.query.mapId;
  var pathways= await LearningPathways.getAllPaths(mapId);
  pathways = [...new Set(pathways.map(item => item.pathwayId))];
  res.send({
    pathways: pathways
  });  
});

router.get("/fetchResources",Authenticator.ensureAuthenticated, async (req, res) => {
  const pathwayId = req.query.pathwayId;
  var pathways= await LearningPathways.getLearningPathwaysByPathwayId(pathwayId);
  var rsrIds=pathways.map(item => item.resourceId);
  var rsrList=await Resource.getResourceListByResourceId(rsrIds);
  res.send({
    resources: rsrList
  });  
});

router.get("/dataInLocalStorage",Authenticator.ensureAuthenticated, async (req, res) => {
    var action=req.query.action;
    if(action=="remove")
    {
    localStorage.removeItem("clickSource");
    localStorage.removeItem("mapId");
    localStorage.removeItem("pathwayId");  
    }
  else{
    localStorage.setItem("clickSource","viewMapPage");
  }
  res.redirect("/learningMaps/trailerGeneration");
});

router.post("/trailerGeneration",Authenticator.ensureAuthenticated, async (req, res) => {
  var fs1 = require('fs')
  var fileList = [];
  var multerStorage = multer.diskStorage({
    destination: function (req, file, callback) {
      callback(null, path.join(__dirname, "../public/trailer_generation/template_2/data/default/author/"));
    },
    filename: function (req, file, callback) {
      callback(null,file.originalname);
    },
  });
  var multerMultipleUpload = multer({ storage: multerStorage }).array(
    "file",
    1
  );

  await multerMultipleUpload(req, res, async function (err) {
    console.log(req.body.destination);
    res.status(200).json({ status: "ok" });
  });
  
});
module.exports = router;