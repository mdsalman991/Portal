var mongoose = require("mongoose");

var CompetencyMapSchema = mongoose.Schema({
	mapName: {
		type: String,
	},

	mapType: {
		type: String,
	},

	numLevels: {
		type: Number,
	},

	numResources: {
		type: Number,
	},

	numTopics: {
		type: Number,
	},

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

var CompetencyMap = module.exports = mongoose.model('CompetencyMap', CompetencyMapSchema);

module.exports.createCompetencyMap = function (newCompetencyMap) {
	return newCompetencyMap.save();
}

module.exports.deleteCompetencyCMapById = (cMapId) => {
	return CompetencyMap.find({
		_id: cMapId
	}).deleteOne();
};