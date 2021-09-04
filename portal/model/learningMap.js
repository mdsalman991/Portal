var mongoose = require("mongoose");

var LearningMapSchema = mongoose.Schema({
	mapName: {
		type: String,
	},

	mapDescription: {
		type: String,
	},

	cMapId: {
		type: mongoose.Schema.Types.ObjectId,
	},

	mapTags: [String],

	createdBy: {
		type: mongoose.Schema.Types.ObjectId,
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

var LearningMap = module.exports = mongoose.model('LearningMap', LearningMapSchema);

module.exports.createLearningMap = function (newLearningMap) {
	return newLearningMap.save();
}

module.exports.getAllLearningMaps = function () {
	return LearningMap.find().lean().exec();
}

module.exports.getCMapId = function (LearningMapId) {
	return LearningMap.find({
		_id: LearningMapId
	});
}

module.exports.getId = function (cMapId) {
	return LearningMap.find({
		cMapId: cMapId
	})
}

module.exports.getLearningMapById = function (mapId) {
	return LearningMap.find({
		_id: mapId
	}).lean().exec();
}

module.exports.deleteLearningMapById = (mapId) => {
	return LearningMap.find({
		_id: mapId
	}).deleteOne();
};