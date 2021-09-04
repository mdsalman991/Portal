var mongoose = require("mongoose");

var LearningMapResourcesSchema = mongoose.Schema({
  mapId: {
    type: mongoose.Schema.Types.ObjectId,
  },

  // ! : Changed to Object ID 
  // resourceId: { type: String },
  resourceId: {
    type: mongoose.Schema.Types.ObjectId,
  },

  // ! Added resource Name attribute for places that require the resource name from learning maps
  resourceName: {
    type: String,
  },

  locationX: {
    type: Number,
  },

  locationY: {
    type: Number,
  },

  normX: {
    type: Number,
  },

  normY: {
    type: Number,
  },

  presentInCompetency: {
    type: Boolean,
    default: false,
  },

  createdOn: {
    type: Date,
    default: Date.now,
  },

  updatedOn: {
    type: Date,
    default: Date.now,
  },
});

var LearningMapResources = (module.exports = mongoose.model(
  "LearningMapResources",
  LearningMapResourcesSchema
));

module.exports.createLearningMapResources = function (newLearningMapResource) {
  return newLearningMapResource.save();
};

module.exports.getAllLearningMapResources = function (learningMapId) {
  return LearningMapResources.find({
    mapId: learningMapId
  });
  // .lean().exec();
};

module.exports.deleteLearningMapResourcesByMapId = (mapId) => {
  return LearningMapResources.find({
    mapId: mapId
  }).deleteMany();
}