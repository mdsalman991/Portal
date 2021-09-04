var mongoose = require("mongoose");

var CmapTopicsSchema = mongoose.Schema({
	mapId: {
		type: mongoose.Schema.Types.ObjectId,
	},

	topicName: {
		type: String,
	},

	topicVolume: {
		type: Number,
	},

	locationX: {
		type: Number,
	},

	topicClusterId: {
		type: Number,
	},

	topicClusterProbability: {
		type: Number,
	},

	topicType: {
		type: String,
	}

});

var CmapTopics = module.exports = mongoose.model('CmapTopics', CmapTopicsSchema);

module.exports.createCmapTopics = function (newCmapTopic) {
	return newCmapTopic.save();
}

module.exports.deleteCmapTopicsByCMapId = (mapId) => {
	return CmapTopics.find({
		mapId: mapId
	}).deleteMany();
}