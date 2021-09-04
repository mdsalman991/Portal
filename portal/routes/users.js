var express = require("express");
var router = express.Router();
var User = require("../model/user");
var passport = require("passport");
var LocalStrategy = require("passport-local").Strategy;
var flash = require("connect-flash");
var session = require("express-session");

var Authenticator = require("../middleware/authenticator");

router.get("/register", function (req, res) {
  res.render("register", { layout: "layoutUser" });
});

router.get("/login", function (req, res) {
  if (req.user != null || req.user != undefined) {
    res.redirect("./dashboard");
  } else {
    res.render("login", { layout: "layoutUser" });
  }
});

router.post("/register", function (req, res) {
  var name = req.body.name;
  var email = req.body.email;
  var username = req.body.username;
  var role = req.body.role;
  var password = req.body.password;
  var password2 = req.body.password2;

  var errors = [];

  if (!name || !email || !username || !role || !password || !password2) {
    errors.push({ msg: "Please fill in all the fields" });
  }

  if (password != password2) {
    errors.push({ msg: "Passwords do not match" });
  }

  if (errors.length > 0) {
    res.render("register", {
      layout: "layoutUser",
      errors: errors,
      name: name,
      email: email,
      username: username,
      role: role,
      password: password,
      password2: password2,
    });
  } else {
    User.findOne({ username: username }).then((user) => {
      if (user) {
        errors.push({ msg: "Username already exists!!" });
        res.render("register", {
          layout: "layoutUser",
          errors: errors,
          name: name,
          email: email,
          username: username,
          role: role,
          password: password,
          password2: password2,
        });
      } else {
        var newUser = new User({
          name: name,
          email: email,
          username: username,
          role: role,
          password: password,
        });

        User.createUser(newUser, function (err, user) {
          if (err) throw err;
          req.flash("success_msg", "You are now registered and can login");
          res.redirect("login");
        });
      }
    });
  }
});

passport.use(
  new LocalStrategy(function (username, password, done) {
    User.getUserByUsername(username, function (err, user) {
      if (err) throw err;

      if (!user) {
        return done(null, false, { message: "Username is not registered" });
      }

      User.comparePassword(password, user.password, function (err, isMatch) {
        if (err) throw err;

        if (isMatch) {
          return done(null, user);
        } else {
          return done(null, false, { message: "Invalid Password" });
        }
      });
    });
  })
);

passport.serializeUser(function (user, done) {
  done(null, user.id);
});

passport.deserializeUser(function (id, done) {
  User.getUserById(id, function (err, user) {
    done(err, user);
  });
});

router.post("/login", function (req, res, next) {
  passport.authenticate("local", {
    successRedirect: "dashboard",
    failureRedirect: "login",
    failureFlash: true,
  })(req, res, next);
});

router.get(
  "/dashboard",
  Authenticator.ensureAuthenticated,
  function (req, res) {
    res.render("dashboard", {
      layout: "layoutDashboard",
      username: req.user.username,
    });
  }
);

router.get("/logout", (req, res) => {
  req.logout();
  req.flash("success_msg", "You are logged out");
  res.redirect("/");
});

module.exports = router;
