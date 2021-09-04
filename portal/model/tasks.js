const mongoose = require("mongoose");

const TaskSchema = mongoose.Schema({

  taskType: {
    type: String,
    enum: ['create_map'],
    required: true,
  },

  taskId: {
    type: String,
  },

  taskStatus: {
    type: String,
    enum: ["SUCCESS", "FAILURE", "STARTED", "PENDING"],
    default: 'STARTED',
    required: true,
  },

  // TODO : Find a better name for this attribute
  // ** This corresponds to the ID of the object in question i.e., it can be a learning map, a pathway, etc.
  objectId: {
    type: mongoose.Schema.Types.ObjectId,
  },
});

const Task = mongoose.model("Task", TaskSchema);

module.exports = Task;

module.exports.createTask = (newTask, callback) => {
  newTask.save(callback);
};

module.exports.findTaskByObjectId = ({id}) => {
  return Task.findOne({objectId : id}).exec();
}

module.exports.updateTaskByTaskId = ({
  taskId,
  taskStatus
}) => {
  return Task.findOneAndUpdate(taskId, {
    taskStatus: taskStatus
  }, {
    new: true
  });
};

module.exports.findTaskbyTaskId = ({
  taskId
}) => {
  return Task.findOne({
    taskId: taskId
  }).exec();
};

module.exports.deleteTaskByObjectId = (objectId) => {
	return Task.find({
		objectId : objectId
	}).deleteOne();
};