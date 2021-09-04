var mongoose = require('mongoose');
const Answer = require('./answers');
const Comment = require('./comments');
const Schema = mongoose.Schema;

var discussionSchema = new Schema({
    dtype: String,
    res_id: String,
    title: String,
    question: String,
    votes: [mongoose.Schema.Types.ObjectId],
    username: String,
    userid: String,
    timestamp: Number,
    answers: [
        {
            type: Schema.Types.ObjectId,
            ref: 'Answer'
        }
    ],
    comments: [
        {
            type: Schema.Types.ObjectId,
            ref: 'Comment'
        }
    ]
},{
    timestamps: true
  });

module.exports = mongoose.model('Discussion', discussionSchema);

