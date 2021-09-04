var mongoose = require("mongoose");

var LearningMapUsersSchema = mongoose.Schema({
	mapId: {
		type: mongoose.Schema.Types.ObjectId,
	},

	userId: {
		type: mongoose.Schema.Types.ObjectId,
	},

	locationX: {
		type: Number,
	},

	locationY: {
		type: Number,
	},

});

var LearningMapUsers = module.exports = mongoose.model('LearningMapUsers', LearningMapUsersSchema);

module.exports.createLearningMapUsers = function (newLearningMapUser) {
	return newLearningMapUser.save();
}

module.exports.getLearningMapUsers = function (learningMapId) {
	return LearningMapUsers.find({
		mapId: learningMapId
	}).lean().exec();
}

module.exports.getLearningMapsByUser = function (userId) {
	return LearningMapUsers.find({
		userId: userId
	}).lean().exec();
}

module.exports.deleteLearningMapUsersByMapId = (mapId) => {
	return LearningMapUsers.find({
		mapId: mapId
	}).deleteMany();
}