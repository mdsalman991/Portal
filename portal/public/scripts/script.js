exports.isauthenticated = function isAuthenticated(req, res, next) {
  // do any checks you want to in here
  // CHECK THE USER STORED IN SESSION FOR A CUSTOM VARIABLE
  // you can do this however you want with whatever variables you set up
  // if (req.user.authenticated)
  //     return next();

  return next();
  res.redirect("/");
};
