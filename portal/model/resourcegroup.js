var mongoose = require('mongoose');

var ResourceGroupSchema = mongoose.Schema({
	resourceGroupName : {
		type: String,
	},

	resourceGroupDescription : {
		type: String,
	},

	resourceGroupTags : [{
		tagName : {
			type: String,
		}
	}],

	resources : [{
		resourceId : {
			type: mongoose.Schema.Types.ObjectId,
		}
	}],

	attentionTime : {
		type: Number,
	},

	createdBy : {
		type: mongoose.Schema.Types.ObjectId,
	},

	createdAt : {
		type: Date,
		default: Date.now
	},

	updatedAt : {
		type: Date,
		default: Date.now
	},

});

var ResourceGroup = module.exports = mongoose.model('ResourceGroup', ResourceGroupSchema);


module.exports.createResourceGroup = function(newResourceGroup){
	return newResourceGroup.save();
} 

module.exports.getAllResourceGroup = function(){
	return ResourceGroup.find().lean().exec();
}


