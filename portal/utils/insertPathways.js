const neatCsv = require("neat-csv");
const fs = require("fs");
var mongoose = require("mongoose");
var ObjectId = require("mongodb").ObjectID;
var LearningPathways = require("../model/learningPathways");
var resources = require("../model/resource");

const appConfig = require("../config/app_config");
const MONGO_CONNECTION_URL = `mongodb://${appConfig.MONGO_HOST}/${appConfig.MONGO_DB}`;
mongoose.connect(MONGO_CONNECTION_URL);


const args = process.argv.slice(2)

// learning-pathways
fs.readFile(
    args[0],
    async (err, data) => {
        if (err) {
            console.error(err);
            return;
        }
        var neatDataList = await neatCsv(data);
        //console.log(neatDataList);
        for (var i = 0; i < neatDataList.length; i++) {
            var neatData = neatDataList[i];
            //console.log(neatData["Pathway LOD"]);
            var obj = new LearningPathways({
                mapId: ObjectId(neatData.mapId),
                pathwayId: neatData.pathwayId,
                resourceId: (await resources.getResourceByResourceName(neatData.resourceName))._id,
                resourceName: neatData.resourceName,               
                sequenceId: neatData.sequenceId,
                pathwayLod: neatData.pathwayLod,
            });
            var created_data = await LearningPathways.createLearningPathways(obj);
            //console.log("dt: ", created_data);
        }
    }
);
console.log("Insertion Complete..!!")