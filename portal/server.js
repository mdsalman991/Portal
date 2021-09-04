//libraries used in our code

var express = require("express");
var bodyParser = require("body-parser");
var path = require("path");
var fs = require("fs");
var cors = require("cors");
var session = require("express-session");
var expressValidator = require("express-validator");
var expressLayouts = require("express-ejs-layouts");
var cookieParser = require("cookie-parser");
var flash = require("connect-flash");
var passport = require("passport");
var mongoose = require("mongoose");
const MongoStore = require("connect-mongo");

const appConfig = require("./config/app_config");
const MONGO_CONNECTION_URL = `mongodb://${appConfig.MONGO_HOST}/${appConfig.MONGO_DB}`;
mongoose.connect(MONGO_CONNECTION_URL,{ useNewUrlParser: true,  useUnifiedTopology: true });

var routes = require("./routes/index");
var users = require("./routes/users");
var resources = require("./routes/resources");
var learningMaps = require("./routes/learningMaps");
const log = require("./routes/log");
const tasks = require("./routes/tasks");
var discussions = require("./routes/discussion");


var app = express();
app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json());
app.use(express.json());
app.use(express.urlencoded({ extended: false }));
app.use(cookieParser());
app.use(expressValidator());
app.use(expressLayouts);
app.set("views", path.join(__dirname, "views"));
app.set("view engine", "ejs");

app.use(
  session({
    secret: "keyboard cat",
    resave: false,
    saveUninitialized: true,
    store: MongoStore.create({
      mongoUrl: MONGO_CONNECTION_URL,
    }),
  })
);

app.use(passport.initialize());
app.use(passport.session());

// Connect flash
app.use(flash());
app.use(function (req, res, next) {
  res.locals.success_msg = req.flash("success_msg");
  res.locals.error_msg = req.flash("error_msg");
  res.locals.error = req.flash("error");
  next();
});

//run the project on localhost at port 5555
app.use(cors({ origin: "http://localhost:5555" }));

app.use("/", routes);
app.use("/users", users);
app.use("/resources", resources);
app.use("/learningMaps", learningMaps);
app.use("/log", log);
app.use("/tasks", tasks);
app.use("/discussions", discussions);

//control shifted to the public directory
app.use(express.static(path.join(__dirname, "public")));

// assign port 5555 to the server
var server = app.listen(5555, function () {
  console.log("Node server is running !!..");
  console.log("Browser to http://127.0.0.1:5555");
});
