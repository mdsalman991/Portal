var mongoose = require("mongoose");

var ResourceSchema = mongoose.Schema({
  resourceName: {
    type: String,
  },

  resourceDescription: {
    type: String,
  },

  resourceLocation: {
    originalLocation: {
      type: String,
    },
    convertedLocation: {
      type: String,
    },
  },

  resourceType: {
    type: String,
    enum: ["Text", "Video", "Both", "Pdf"],
    default: "Text",
    required: true,
  },

  resourceTags: [{
    tagName: {
      type: String,
    },
  }, ],

  attentionTime: {
    type: Number,
    default: 10,
  },

  uploadedBy: {
    type: mongoose.Schema.Types.ObjectId,
  },

  likedBy: [mongoose.Schema.Types.ObjectId],

  dislikedBy: [mongoose.Schema.Types.ObjectId],

  completedBy: [mongoose.Schema.Types.ObjectId],

  createdAt: {
    type: Date,
    default: Date.now,
  },

  updatedAt: {
    type: Date,
    default: Date.now,
  },
});

var Resource = (module.exports = mongoose.model("Resource", ResourceSchema));

module.exports.createResource = function (newResource) {
  return newResource.save();
};

module.exports.deleteResource = function(resourceId){
	return Resource.findOne({ _id: resourceId }).deleteOne();
}

module.exports.editResource = function(resourceId,data){
	return Resource.findOne({ _id: resourceId }).updateOne(data);
}

module.exports.getResourceByResourceId = function (resourceID) {
  return Resource.findOne({
    _id: resourceID
  }).exec();
};

module.exports.getResourceByResourceName = function (resourceName) {
  return Resource.findOne({
    resourceName: resourceName
  }).exec();
};

module.exports.getResourceByResourceNameRegex = function (resourceName) {
  return Resource.findOne({
    resourceName: {
      $regex: ".*" + resourceName + ".*"
    },
  }).exec();
};

module.exports.getAllResource = function () {
  return Resource.find().lean().exec();
};

module.exports.getResourceListByResourceNames = function (resourceName) {
  return Resource.find({
    resourceName: {
      $in: resourceName
    }
  });
};

module.exports.getResourceListByResourceId = function (resourceIds) {
  return Resource.find({
    _id: {
      $in: resourceIds
    }
  });
};

module.exports.getAttentionTime = async function (resourceId) {
  // ? This is should either be resource.resourceId or be replaced with only getting resourceId as the argument rather than the resource object
  var id = resourceId.resourceId;

  return Resource.findById(id).exec();
};

module.exports.updateLikes = async function (rsrId, userId, action) {
  if (action === "like") {
    return Resource.findByIdAndUpdate(
      rsrId, {
        $addToSet: {
          likedBy: userId
        }
      }, {
        new: true,
        upsert: true
      }
    ).exec();
  } else {
    return Resource.findByIdAndUpdate(
      rsrId, {
        $pull: {
          likedBy: userId
        }
      }, {
        new: true,
        upsert: true
      }
    ).exec();
  }
};

module.exports.updateDislikes = async function (rsrId, userId, action) {
  if (action === "dislike") {
    return Resource.findByIdAndUpdate(
      rsrId, {
        $addToSet: {
          dislikedBy: userId
        }
      }, {
        new: true,
        upsert: true
      }
    ).exec();
  } else {
    return Resource.findByIdAndUpdate(
      rsrId, {
        $pull: {
          dislikedBy: userId
        }
      }, {
        new: true,
        upsert: true
      }
    ).exec();
  }
};

module.exports.updateCompleted = async function (rsrId, userId) {
  return Resource.findByIdAndUpdate(
    rsrId, {
      $addToSet: {
        completedBy: userId
      }
    }, {
      new: true,
      upsert: true
    }
  ).exec();
};

module.exports.getFileType = ({
  resourceIds
}, callback) => {
  // ! Changed to _id from resourceName
  return Resource.find({
      _id: resourceIds
    },
    "resourceName resourceType",
    callback
  );
};