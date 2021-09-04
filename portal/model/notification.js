var mongoose = require('mongoose');

var NotificationSchema = mongoose.Schema({
	username : {
		type: mongoose.Schema.Types.ObjectId,
	},

	message : 
    {
		type: String,
		required: true
	}
});

var Notification = module.exports = mongoose.model('Notification', NotificationSchema);


module.exports.AddNotification = function(newNotification){
	return newNotification.save();
} 

module.exports.getNotification = function(ID){
	return Notification.find( { username: ID },{_id:0,username:0,__v:0}).exec();
}