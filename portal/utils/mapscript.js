const neatCsv = require("neat-csv");
const fs = require("fs");
var mongoose = require("mongoose");
var ObjectId = require("mongodb").ObjectID;
var CompetencyMap = require("../model/competencyMap");
var CMapResources = require("../model/cmapResources");
var CMapTopics = require("../model/cmapTopics");
var CMapResourceTopics = require("../model/cmapResourceTopics");
var LearningMap = require("../model/learningMap");
var LearningMapResources = require("../model/learningMapResources");
var LearningPathways = require("../model/learningPathways");

const appConfig = require("../config/app_config");
const MONGO_CONNECTION_URL = `mongodb://${appConfig.MONGO_HOST}/${appConfig.MONGO_DB}`;
mongoose.connect(MONGO_CONNECTION_URL);

var user_id = "5fd60a9d9d68890ea328f435";
var mapFileLocation =  "/c/Users/vnshy/Documents/learning_maps/portal/nlp_maps_top2vec";
var competencyMapName = "NLP TOP2VEC1";
var learningMapName = "NLP-Top2Vec";
var mapDescription = "NLP maps generated using top2vec topic model";
var numResources = "";
var numTopics = "";

var complete = 0;
// competency map
var competency_map = new CompetencyMap({
  mapName: competencyMapName,
  mapType: "Text",
  numLevels: 10,
  numResources: numResources,
  numTopics: numTopics,
  createdBy: ObjectId(user_id),
});

CompetencyMap.createCompetencyMap(competency_map).then(function (d) {
  var cmap_id = d._id;
  // cmap-resources

  fs.readFile(mapFileLocation + "/cmaps/resources.csv", async (err, data) => {
    if (err) {
      console.error(err);
      return;
    }
    var neatDataList = await neatCsv(data);

    for (var i = 0; i < neatDataList.length; i++) {
      var neatData = neatDataList[i];
      var obj = new CMapResources({
        mapId: ObjectId(cmap_id),
        resourceId: neatData.resource_id,
        locationX: neatData.X,
        locationY: neatData.Y,
        normX: neatData.norm_X,
        normY: neatData.norm_Y,
        //   zoom: neatData.zoom,
      });

      var created_data = await CMapResources.createCmapResources(obj);
    }
  });

  // cmap-topics
  fs.readFile(mapFileLocation + "/cmaps/topics.csv", async (err, data) => {
    if (err) {
      console.error(err);
      return;
    }
    var neatDataList = await neatCsv(data);

    for (var i = 0; i < neatDataList.length; i++) {
      var neatData = neatDataList[i];
      var obj = new CMapTopics({
        mapId: ObjectId(cmap_id),
        topicName: neatData.topic_name,
        topicVolume: neatData.topic_value,
        locationX: neatData.X,
        topicClusterId: neatData.topic_cluster_id,
        topicClusterProbability: neatData.topic_cluster_probability,
        topicType: neatData.topic_type,
      });

      var created_data = await CMapTopics.createCmapTopics(obj);
      //console.log("dt: ", created_data);
    }
  });

  // cmaps-resource-topics
  fs.readFile(
    mapFileLocation + "/cmaps/resource_mapping.csv",
    async (err, data) => {
      if (err) {
        console.error(err);
        return;
      }
      var neatDataList = await neatCsv(data);
      //console.log(neatDataList);

      for (var i = 0; i < neatDataList.length; i++) {
        var neatData = neatDataList[i];
        var obj = new CMapResourceTopics({
          mapId: ObjectId(cmap_id),
          resourceId: neatData.resource_id,
          topicName: neatData.topic_name,
          resourceMappedProbability: neatData.document_mapped_probability,
        });

        var created_data = await CMapResourceTopics.createCmapResourceTopics(
          obj
        );
        //console.log("dt: ", created_data);
      }
    }
  );

  // learning map
  var learning_map = new LearningMap({
    mapName: learningMapName,
    mapDescription: mapDescription,
    cMapId: ObjectId(cmap_id),
    mapTags: [],
    createdBy: ObjectId(user_id),
  });

  LearningMap.createLearningMap(learning_map).then(function (d) {
    var learning_map_id = d._id;
    // LearningMap.getId(ObjectId("5f92fa075c9bbeb3b03bacc0"))
    console.log(learning_map_id);
    fs.readFile(mapFileLocation + "/cmaps/resources.csv", async (err, data) => {
      if (err) {
        console.error(err);
        return;
      }
      var neatDataList = await neatCsv(data);

      for (var i = 0; i < neatDataList.length; i++) {
        var neatData = neatDataList[i];
        var obj = new LearningMapResources({
          mapId: ObjectId(learning_map_id),
          resourceId: neatData.resource_id,
          locationX: neatData.X,
          locationY: neatData.Y,
          normX: neatData.norm_X,
          normY: neatData.norm_Y,
          presentInCompetency: true,
        });

        var created_data = await LearningMapResources.createLearningMapResources(
          obj
        );
        //console.log("dt: ", created_data);
      }
    });

    // learning-pathways
    fs.readFile(
      mapFileLocation + "/pathways/pathways.csv",
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
            mapId: ObjectId(learning_map_id),
            resourceId: neatData.resource_id,
            pathwayId: neatData.Pathway_ID,
            pathwayLod: neatData.Pathway_LOD,
            sequenceId: neatData.sequence_id,
          });

          var created_data = await LearningPathways.createLearningPathways(obj);
          //console.log("dt: ", created_data);
        }
      }
    );
  });
  console.log("NLP Map added");
});
console.log("All Done...");
