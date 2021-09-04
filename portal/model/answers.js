var mongoose = require('mongoose');
const Comment = require('./comments');
const Schema = mongoose.Schema;

var answerSchema = new Schema({
    answer: String,
    votes: [mongoose.Schema.Types.ObjectId],
    username: String,
    userid: String,
    timestamp: Number,
    TA_verified: { type: Boolean, default: false, },
    TA_intervene: { type: Boolean, default: false, },
    no_intervene_requested: { type: Number, default: 0, },
    Branch_result: { type: String, default: 'none', },
    thread_close: { type: Boolean, default: false, },
    inactive:{ type: Boolean, default: false, },
    comments: [
        {
            type: Schema.Types.ObjectId,
            ref: 'Comment'
        }
    ]
},{
    // Make Mongoose use Unix time (seconds since Jan 1, 1970)
    timestamps: true
  });

module.exports = mongoose.model('Answer', answerSchema);
