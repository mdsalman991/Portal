var mongoose = require("mongoose");

var CmapResourcesSchema = mongoose.Schema({
  mapId: {
    type: mongoose.Schema.Types.ObjectId
  },

  // ! Changed to Object ID
  // resourceId: { type: String },
  resourceId: {
    type: mongoose.Schema.Types.ObjectId
  },

  locationX: {
    type: Number
  },

  locationY: {
    type: Number
  },

  normX: {
    type: Number
  },

  normY: {
    type: Number
  },
});

var CmapResources = (module.exports = mongoose.model(
  "CmapResources",
  CmapResourcesSchema
));

module.exports.createCmapResources = function (newCmapResource) {
  return newCmapResource.save();
};

module.exports.getAllResources = function (learningMapId) {
  return CmapResources.find({
    mapId: learningMapId,
  });
  // .lean().exec()
};

module.exports.deleteCmapResourcesByCMapId = (mapId) => {
  return CmapResources.find({
    mapId: mapId
  }).deleteMany();
}