var mongoose = require("mongoose");

var CmapResourceTopicsSchema = mongoose.Schema({
  mapId: {
    type: mongoose.Schema.Types.ObjectId
  },

  // ! : Changed to Object ID
  // resourceId: { type: String },
  resourceId: {
    type: mongoose.Schema.Types.ObjectId
  },

  topicName: {
    type: String
  },

  resourceMappedProbability: {
    type: Number
  },
});

var CmapResourceTopics = (module.exports = mongoose.model(
  "CmapResourceTopics",
  CmapResourceTopicsSchema
));

module.exports.createCmapResourceTopics = function (newCmapResourceTopic) {
  return newCmapResourceTopic.save();
};

module.exports.getAllTopics = function (learningMapId) {
  return CmapResourceTopics.find({
    mapId: learningMapId,
  });
};

module.exports.deleteCmapResourceTopicsByCMapId = (mapId) => {
  return CmapResourceTopics.find({
    mapId: mapId
  }).deleteMany();
}