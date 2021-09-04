var mongoose = require('mongoose');

var commentSchema = new mongoose.Schema({
    comment: String,
    username: String,
    userid: String,
    timestamp: Number,
    annotate: { type: String, default: "None", },
    TA_verified: { type: Boolean, default: false, },
    votes: [mongoose.Schema.Types.ObjectId]
},{
    // Make Mongoose use Unix time (seconds since Jan 1, 1970)
    timestamps: true
  });

module.exports = mongoose.model('Comment', commentSchema);
