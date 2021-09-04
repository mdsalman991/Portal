var mongoose = require("mongoose");

var LearningPathwaysSchema = mongoose.Schema({
	mapId: {
		type: mongoose.Schema.Types.ObjectId,
	},

	pathwayId: {
		type: String,
	},

	// ! : Changed to Object ID 
	// resourceId: { type: String, },
	resourceId: {
		type: mongoose.Schema.Types.ObjectId,
	},

	// ! Added resource Name attribute for places that require the resource name from learning pathways
	resourceName: {
		type: String,
	},


	sequenceId: {
		type: Number,
	},

	pathwayLod: {
		type: Number,
	}

});

var LearningPathways = module.exports = mongoose.model('LearningPathways', LearningPathwaysSchema);

module.exports.createLearningPathways = function (newLearningPathway) {
	return newLearningPathway.save();
}


module.exports.getAllPaths = function (learningMapId) {
	return LearningPathways.find({
		mapId: learningMapId
	});
}

module.exports.getLearningPathwaysByMapId = function (mapId) {
	return LearningPathways.find(({
		mapId: mapId
	})).lean().exec();
}

module.exports.getLearningPathwaysByPathwayId = function (pathwayId) {
	return LearningPathways.find(({
		pathwayId: pathwayId
	})).lean().exec();
}

module.exports.deleteLearningPathwaysByMapId = (mapId) => {
	return LearningPathways.find({
		mapId: mapId
	}).deleteMany();
}