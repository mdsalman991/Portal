const mongoose = require("mongoose");

const LogSchema = mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
  },
  learningMapId: {
    type: mongoose.Schema.Types.ObjectId,
  },
  learningMapResourceId: {
    type: mongoose.Schema.Types.ObjectId,
  },
  logTime: {
    type: Date,
    default: Date.now,
  },
});

const Log = mongoose.model("Log", LogSchema);

module.exports = Log;

module.exports.createLog = (newLog, callback) => {
  newLog.save(callback);
};

module.exports.findLastNLogForUser = (userId, N) =>
  Log.find({ userId: userId }).sort({ logTime: -1 }).limit(N);

module.exports.findResourceLog = ({ resourceIds }, callback) => {
  Log.find(
    {
      learningMapResourceId: { $in: resourceIds },
    },
    callback
  );
};
