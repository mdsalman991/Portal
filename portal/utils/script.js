var mongoose = require("mongoose");
var User = require("../model/user");
const appConfig = require("../config/app_config");
const MONGO_CONNECTION_URL = `mongodb://${appConfig.MONGO_HOST}/${appConfig.MONGO_DB}`;
mongoose.connect(MONGO_CONNECTION_URL);

var dummy_user = new User({
  name: "dummy_user",
  email: "dummy_user@gmail.com",
  username: "dummy_user",
  role: "user",
  password: "qw",
});

var dummy_teacher = new User({
  name: "dummy_teacher",
  email: "dummy_teacher@gmail.com",
  username: "dummy_teacher",
  role: "teacher",
  password: "qw",
});

var dummy_admin = new User({
  name: "dummy_admin",
  email: "dummy_admin@gmail.com",
  username: "dummy_admin",
  role: "admin",
  password: "qw",
});

User.createUser(dummy_user, function (err, user) {
  if (err) throw err;
});

User.createUser(dummy_teacher, function (err, user) {
  if (err) throw err;
});

User.createUser(dummy_admin, function (err, user) {
  if (err) throw err;
});
