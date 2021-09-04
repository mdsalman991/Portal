var express = require("express");
var router = express.Router();
var path = require("path");
const multer = require("multer");
const neatCsv = require("neat-csv");
var LearningPathways = require("../model/learningPathways");

var fs = require("fs").promises;
var ObjectId = require("mongodb").ObjectID;
const pdfParse = require("pdf-parse");

var User = require("../model/user");
var notification = require("../model/notification");
var ResourceGroup = require("../model/resourcegroup");
var Resource = require("../model/resource");
var Authenticator = require("../middleware/authenticator");

const { getVideoDurationInSeconds } = require("get-video-duration"); //npm -i get-video-duration
const youtubedl = require("ytdl-core"); //npm install youtube-dl;npm install ytdl-core
const getSubtitles = require('youtube-captions-scraper').getSubtitles;
const sizeOf = require('request-image-size');

const APP_CONFIG = require("../config/app_config");
const EventEmitter = require("events");
const { set } = require("mongoose");
const emitter = new EventEmitter();
const RAW_LOCATION = APP_CONFIG.RAW_LOCATION;
const CONVERTED_LOCATION = APP_CONFIG.CONVERTED_LOCATION;

const convertedLocation = path.join(__dirname, CONVERTED_LOCATION);
var re = /(?:\.([^.]+))?$/;

const unique = (value, index, self) => {
  return self.indexOf(value) === index;
};

const calculateAttentionTime = async function (filename) {
  let destination = path.join(__dirname, RAW_LOCATION, filename);
  let data = await fs.readFile(destination);
  let datastring = data.toString();
  datastring = datastring.split(" ");
  let noOfWords = datastring.length;
  return Math.ceil(noOfWords / 200);
};

const calculateAttentionTimePDF = async function (filename) {
  let destination = path.join(__dirname, RAW_LOCATION, filename);
  let dataBuffer = await fs.readFile(destination);
  let data = await pdfParse(dataBuffer);
  let pdfString = data.text;
  var convertedFilename = filename.split(".")[0];
  convertedFilename += ".txt";
  let convertedFileDestination = path.join(
    __dirname,
    CONVERTED_LOCATION,
    convertedFilename
  );
  let wriitenText = await fs.writeFile(convertedFileDestination, pdfString);
  pdfString = pdfString.split(" ");
  let noOfWords = pdfString.length;
  return Math.ceil(noOfWords / 200);
};

const calculateAttentionTimeVideo = async function (filename) {
  let destination = path.join(__dirname, RAW_LOCATION, filename);
  let fs = require("fs");
  let dataBuffer = await fs.createReadStream(destination);
  return getVideoDurationInSeconds(dataBuffer);
};

const getPdfString = async function (destination) {
  let dataBuffer = await fs.readFile(destination);
  let data = await pdfParse(dataBuffer);
  let pdfString = data.text;
  pdfString = pdfString.toString("base64");
  return pdfString;
};

const getTextFile = async function (destination) {
  let data = await fs.readFile(destination);
  return data;
};

var multerStorage = multer.diskStorage({
  destination: function (req, file, callback) {
    callback(null, path.join(__dirname, RAW_LOCATION));
  },
  filename: function (req, file, callback) {
    callback(null, Date.now() + "_" + file.originalname);
  },
});

router.get(
  "/notification",
  Authenticator.ensureAuthenticated,
  async (req, res) => {
    var noti = await notification.getNotification(req.user._id);
    res.send({ noti: noti });
  }
);

router.post("/addnotification", async (req, res) => {
  var newNoti = new notification({
    username: req.user._id,
    message: req.body.message,
  });
  var notistatus = await notification.AddNotification(newNoti);
  res.status(200).json({ status: "ok" });
});
router.post("/addTAnotification", async (req, res) => {
  users = await User.find({});
  console.log(users.length)
  for(var i = 0;i<users.length;i++){
    if(users[i].role == 'TA'){
      var newNoti = new notification({
        username: users[i]._id,
        message: req.body.message
      });
      var noti = await notification.getNotification(users[i]._id);
      if(!noti.some(e => e.message == req.body.message)) {
        var notistatus = await notification.AddNotification(newNoti);
      }
    }
  }
  res.status(200).json({ status: "ok" });
});


router.get("/viewResourceGroup", Authenticator.ensureAuthenticated, async (req, res) => {
  var resourceGroupList = await ResourceGroup.getAllResourceGroup();
  
  var resourceList = await Resource.getAllResource();


  let resourceListString = "";

  for (let i = 0; i < resourceList.length; i++) {
    if (i > 0) {
      resourceListString += "|";
    }
    resourceListString += JSON.stringify(resourceList[i]);
  }

  for (let j = 0; j < resourceGroupList.length; j++) {
    let rsrGrpList = resourceGroupList[j].resources;
    let totTime = 0;
    for (let i = 0; i < rsrGrpList.length; i++) {
      let rsrId = rsrGrpList[i];
      let rsr = await Resource.getAttentionTime(rsrId);
      totTime += rsr.attentionTime;
    }
    
    resourceGroupList[j].totalAttentionTime = totTime;
    var createdById = resourceGroupList[j].createdBy;
    // BELOW LINE MAY NOT WORK IF USER FOR GIVEN _ID DOESN'T EXIST
    var createdByUsername = await User.getUserByUserId(ObjectId(createdById));
    resourceGroupList[j].createdByUsername = createdByUsername[0].username;
  }

  for (let j = 0; j < resourceGroupList.length; j++) {
    let resourceIdsString = "";
    let rsrGrpList = resourceGroupList[j].resources;
    for (let i = 0; i < rsrGrpList.length; i++) {
      if (i > 0) {
        resourceIdsString += "|";
      }
      resourceIdsString += JSON.stringify(rsrGrpList[i]);
    }
    resourceGroupList[j].resourceIdsString = resourceIdsString;
  }

  res.render("viewResourceGroup", {
    layout: "layoutViewResourceGroup",
    user: req.user,
    username: req.user.username,
    resourceList: resourceList,
    resourceGroupList: resourceGroupList,
    resourceListString: resourceListString,
  });
});

router.get(
  "/viewResource",
  Authenticator.ensureAuthenticated,
  async (req, res) => {
    var resourceList = await Resource.getAllResource();
    for (let i = 0; i < resourceList.length; i++) {
      let resource = resourceList[i];
      var createdById = resource.uploadedBy;
      // BELOW LINE MAY NOT WORK IF USER FOR GIVEN _ID DOESN'T EXIST
      var createdByUsername = await User.getUserByUserId(ObjectId(createdById));
      resource.createdByUsername = createdByUsername[0].username;
    }

    res.render("viewResource", {
      layout: "layoutViewResource",
      user: req.user,
      username: req.user.username,
      resourceList: resourceList,
    });
  }
);

router.post("/viewResource", async (req, res) => {
  var fileList = [];
  var fs1 = require('fs')
  var multerMultipleUpload = multer({ storage: multerStorage }).array(
    "multipleFiles",
    500
  );

  await multerMultipleUpload(req, res, async function (err) {
    fileList = req.files;

    if (err) {
      return res.end("Files upload unsucessfull!");
    }
    var uploadType=req.body.uploadType;
    //'0' means upload type is file upload
    if(uploadType=='0')
    {

      req.app.locals.uploadStatus = true;
    
      const resourceTagsString = req.body.updatedResourceTags;
      
      const updatedFileNamesString = req.body.updatedFileNames;
      
      const updatedResourceDescriptionsString = req.body.updatedResourceDescriptions;
      
      const updatedFileNames = updatedFileNamesString.split(",");
      
      const updatedResourceDescriptions = updatedResourceDescriptionsString.split(",");
      
      const updatedResourceTags = resourceTagsString.split(",");
      
      var resourceIds = [];
    
      var fileListArray = Array.from(fileList);

      for (let i = 0; i < fileListArray.length; i++) {
        var resourceTagsList = [];
        
        var eachResourceTags = updatedResourceTags[i];
        
        resourceTagsList.push({
          tagName: eachResourceTags,
        });
    
        var file = fileListArray[i];
      
        var ext = re.exec(file.filename)[1];

        let attentionTime = 10;
        
        let rsrType;
        
        if (ext === "pdf") {
          attentionTime = await calculateAttentionTimePDF(file.filename);
          rsrType = "Pdf";
        }
        
        else if (ext === "txt") {
          attentionTime = await calculateAttentionTime(file.filename);
          rsrType = "Text";
        }
        
        else if (ext === "mp4") {
          attentionTime = await calculateAttentionTimeVideo(file.filename);
          attentionTime = attentionTime/60;
          attentionTime= attentionTime.toFixed(2);
          rsrType = "Video";
        }

        var filenameWithoutExt = file.filename;
        
        filenameWithoutExt = filenameWithoutExt.split(".")[0];

        var pathToConverted = path.join(convertedLocation, filenameWithoutExt);
        
        var newResource = new Resource({
          resourceName: updatedFileNames[i],
          resourceDescription: updatedResourceDescriptions[i],
          resourceLocation: {
            originalLocation: file.path,
            convertedLocation: pathToConverted,
          },
          resourceType: rsrType,
          attentionTime: attentionTime,
          resourceTags: resourceTagsList,
          uploadedBy: req.user._id,
        });
        
        var rsr = await Resource.createResource(newResource);
        
        resourceIds.push({
          resourceId: rsr._id,
        });
      }
      
      res.status(200).json({ status: "ok" });
    
    }
    //'1' means upload type is url upload
    else if(uploadType=='1')
    { 
      let destination = path.join(__dirname, RAW_LOCATION);
      
      let urls=req.body.urlValues;
      
      let updatedResourceTags = req.body.updatedResourceTags;
      
      let updatedResourceDescriptions = req.body.updatedResourceDescriptions;
      
      let urlFileNames=req.body.urlFileNames;
      
      urls=urls.split(",");
      
      urlFileNames = urlFileNames.split(",");
      
      updatedResourceDescriptions = updatedResourceDescriptions.split(",");
      
      updatedResourceTags = updatedResourceTags.split(",");
      
      var resourceIds = [];
      
      for (let i = 0; i < urls.length; i++) {
        
        let url=urls[i];
        
        let urlFileName=urlFileNames[i];
        
        urlFileName+=".mp4";
        
        let videoPath=destination+'/'+urlFileName;
        
        youtubedl(url, {filter: 'audioandvideo', quality: 'highestaudio'},{format: 'mp4'}).pipe(fs1.createWriteStream(videoPath));
        
        let attentionTime = 10;
        
        let rsrType;
        
        let info = await youtubedl.getInfo(youtubedl.getURLVideoID(url));
      
        getSubtitles({
          videoID: youtubedl.getURLVideoID(url),
          lang: 'en' 
        }).then(function(captions) {
          
          let fs1=require('fs');
          
          let dest=pathToConverted+'.txt';
          
          for (let i = 0; i < captions.length; i++) {
            fs1.appendFile(dest, captions[i].text, err => {
              
                if (err) {
                //console.error(err)
                }
            })
          }
        })
        .catch(err=>console.log("can't get transcript..."))
        ;

        attentionTime = info.videoDetails.lengthSeconds;
      
        attentionTime = attentionTime/60;
      
        attentionTime= attentionTime.toFixed(2);
      
        rsrType = "Video";
      
        var filenameWithoutExt=urlFileName.split(".")[0];
      
        var pathToConverted = path.join(convertedLocation, filenameWithoutExt);
      
        var resourceTagsList = [];
        
        resourceTagsList.push({
              tagName: updatedResourceTags[i],
        });
      
        var newResource = new Resource({
          resourceName: urlFileName,
          resourceDescription: updatedResourceDescriptions[i],
          resourceLocation: {
            originalLocation: videoPath,
            convertedLocation: pathToConverted,
          },
          resourceType: rsrType,
          attentionTime: attentionTime,
          resourceTags: resourceTagsList,
          uploadedBy: req.user._id,
        });
      
        var rsr = await Resource.createResource(newResource);
        resourceIds.push({
          resourceId: rsr._id,
        });
      }    
      res.status(200).json({ status: "ok" }); 
    }
    //'2' is for checking whether the url is valid youtube url or not.
    //Note: uploadType is just used as a flag to perform respective actions by server side.
    else if(uploadType=='2')
    {   
        let urls=req.body.urlValues;
        
        urls=urls.split(",");
        
        let f=0;        
        
        for (let i = 0; i < urls.length; i++) { 
          let url=urls[i];  
          try{
            //this checks whether it is youtube url or not
              id=youtubedl.getURLVideoID(url);
              // the below code now checks whether a valid video exists in the youtube url or not.
              var imgsrc = "http://img.youtube.com/vi/" + id + "/mqdefault.jpg";
              
              var size;
              
              await sizeOf(imgsrc).then(data=>size=data.width);
              //it's a trick used to find out whether the youtube url actually has an existing video( if not the thumbnail will have a width of 120)
              
              if(size===120)
              {
                f=1;
              }
            
            }
          catch(err){
              
              f=1;
            }
        
          if(f==1){break;}
        }
        
        if(f==1){res.status(404).json({ status: "Not Found" });}
         
        else{res.status(200).json({ status: "ok" }); }
     }
  
    
  });
});

router.get(
  "/createCollection",
  Authenticator.ensureAuthenticated,
  async (req, res) => {
    
    var resourceList = await Resource.getAllResource();

    res.render("createCollection", {
      layout: "layoutViewResource",
      resourceList: resourceList,
      user: req.user,
      username: req.user.username,
    });
  
  }
);

router.post(
  "/createCollection",
  Authenticator.ensureAuthenticated,
  async (req, res) => {
    const collectionName = req.body.collectionName;
    
    const collectionDescription = req.body.collectionDescription;
    
    const collectionTags = req.body.collectionTags;
    
    const resourceTags = collectionTags.split(",");

    var resourceTagsList = [];

    for (let i = 0; i < resourceTags.length; i++) {
      resourceTagsList.push({
        tagName: resourceTags[i],
      });
    }

    let resourceList;
    
    if (Array.isArray(req.body.resources))
      resourceList = req.body.resources;
    
    else
      resourceList = [req.body.resources];

    const user = req.user;

    var resourceIds = [];
    
    for (let i = 0; i < resourceList.length; i++) {
      
      var rsr= await Resource.getResourceByResourceName(resourceList[i]);
      
      resourceIds.push({
        resourceId: rsr._id,
      });

    }
    var newResourceGroup = new ResourceGroup({
      resourceGroupName: collectionName,
      resourceGroupDescription: collectionDescription,
      resourceGroupTags: resourceTagsList ,
      resources: resourceIds,
      createdBy: user._id,
    });

    var rsrGroup = await ResourceGroup.createResourceGroup(newResourceGroup);
    
    res.redirect("/resources/viewResourceGroup"); 
  }
);

var tresourceList;
router.get(
  "/getSelectedResource",
  Authenticator.ensureAuthenticated,
  async (req, res) => {
    tresourceList = JSON.parse(
      JSON.stringify(
        await Resource.getResourceListByResourceNames(req.query.resourceName)
      )
    );

    const sortedResourceList = [];
    req.query.resourceName.forEach((name) => {
      let resource = tresourceList.filter(
        (tresource) => tresource.resourceName === name
      )[0];

      if (resource === undefined) {} else sortedResourceList.push(resource);
    });
    tresourceList = sortedResourceList;

    for (i in tresourceList) {
      // BELOW LINE MAY NOT WORK IF USER FOR GIVEN _ID DOESN'T EXIST
      uname = await User.getUserByUserId(ObjectId(tresourceList[i].uploadedBy));
      tresourceList[i].uploadedBy = uname[0].username;
    }

    res.json({
      user: req.user,
      username: req.user.username,
      resourceList: tresourceList,
    });
  }
);

router.get(
  "/drawSelectedResource",
  Authenticator.ensureAuthenticated,
  async (req, res) => {
    //in the below code localStorage is used for the purpose of create trailer button in resource list div of view Map page(this is furthur used in trailerGeneration page)
    localStorage.setItem("pathwayId",JSON.stringify(req.query.pathwayId));
    res.render("drawSelectedResource", {
      layout: false,
      user: req.user,
      username: req.user.username,
      resourceList: tresourceList,
    });
  }
);

router.get(
  "/updateLikes",
  Authenticator.ensureAuthenticated,
  async (req, res) => {
    const resourceID = req.query.resource;
  
    const rsr = await Resource.getResourceByResourceId(resourceID);
    const filename = rsr.resourceLocation.originalLocation;

    const action = req.query.action;

    if (action === "like") {
      const updatedResource = await Resource.updateLikes(
        rsr._id,
        req.user._id,
        action
      );
      const resource = await Resource.getResourceByResourceId(rsr._id);
      var noOfLikes = resource.likedBy.length;
      res.send({
        noOfLikes: noOfLikes
      });
    } else {
      const updatedResource = await Resource.updateLikes(
        rsr._id,
        req.user._id,
        action
      );
      const resource = await Resource.getResourceByResourceId(rsr._id);
      var noOfLikes = resource.likedBy.length;
      res.send({
        noOfLikes: noOfLikes
      });
    }
  }
);

router.get(
  "/updateDislikes",
  Authenticator.ensureAuthenticated,
  async (req, res) => {
    const resourceID = req.query.resource;
    const rsr = await Resource.getResourceByResourceId(resourceID);
    const filename = rsr.resourceLocation.originalLocation;

    const action = req.query.action;

    if (action === "dislike") {
      const updatedResource = await Resource.updateDislikes(
        rsr._id,
        req.user._id,
        action
      );
      const resource = await Resource.getResourceByResourceId(rsr._id);
      var noOfDislikes = resource.dislikedBy.length;
      res.send({
        noOfDislikes: noOfDislikes
      });
    } else {
      const updatedResource = await Resource.updateDislikes(
        rsr._id,
        req.user._id,
        action
      );
      const resource = await Resource.getResourceByResourceId(rsr._id);
      var noOfDislikes = resource.dislikedBy.length;
      res.send({
        noOfDislikes: noOfDislikes
      });
    }
  }
);

// when reading a document is completed
router.get(
  "/updateCompleted",
  Authenticator.ensureAuthenticated,
  async (req, res) => {
    const resourceID = req.query.resource;
    const rsr = await Resource.getResourceByResourceId(resourceID);
    const filename = rsr.resourceLocation.originalLocation;

    const updatedResource = await Resource.updateCompleted(
      rsr._id,
      req.user._id
    );
    res.send("done");
  }
);

router.get("/deleteResource", Authenticator.ensureAuthenticated, async (req, res) => {
  
  const resourceID = req.query.resource;
  const rsr = await Resource.getResourceByResourceId(resourceID);
  await Resource.deleteResource(rsr._id);
  try{
    await fs.unlink(rsr.resourceLocation.originalLocation);
    await fs.unlink(rsr.resourceLocation.convertedLocation+'.txt');
  }
  catch(error){/*console.error('error in deleting file',error.message);*/}
  res.send("done");

});

router.post("/editResource", Authenticator.ensureAuthenticated, async (req, res) => {
  
  var multerUpload = multer({ storage: multerStorage }).array("multipleFiles",500);
  await multerUpload(req, res, async function (err) {
  
  const rsr = await Resource.getResourceByResourceId(req.body.resourceID);
  var resourceTagsList = [];
  var ResourceTags = req.body.editedResourceTags;
  resourceTagsList.push({
          tagName: ResourceTags,
        });
  
  const data={ "$set": { "resourceName": req.body.editedResourceName, "resourceDescription":req.body.editedResourceDescription , "resourceTags": resourceTagsList}};
  await Resource.editResource(rsr._id,data);
  res.status(200).json({ status: "ok" });

  });

});

module.exports = router;