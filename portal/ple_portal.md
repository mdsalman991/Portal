# PLE  PORTAL

# Installation

Portal requires [Node.js](https://nodejs.org/) v4+ to run.

Install the dependencies  and start the server.

```sh
$ cd portal
$ npm install
$ node server2.js
```

# Routes

  - `GET /`  landing page
  - `GET /users/register`  register page
  - `POST /users/register`  send the sign up data to server (create user)
  - `GET /users/login` login page
  - `POST /users/login` send the login data to server for verification
  - `GET /dashboard` user dashboard (after login)


# Database
### Downloading MongoDB
[For windows ](https://docs.mongodb.com/manual/tutorial/install-mongodb-on-windows/)
[For Linux](https://docs.mongodb.com/manual/administration/install-on-linux/)
[For MacOS](https://docs.mongodb.com/manual/tutorial/install-mongodb-on-os-x/)

### Create Database

First we create our database `portal`:
```sh
$ use portal
```
### Collections
We have the following collections in our `portal` database:
* `users`
-`name`
-`username`
-`password`
-`email`
-`role`

### Create Dummy Data
We create some dummy data at the start.
It can be created by running the `script` file.
```
$ node script.js
```

This script creates the following documents:
```
{
    name: 'dummy_user',
    username: 'dummy_user',
    email: 'dummy_user@gmail.com',
    role: 'user',
    password: 'qw'
}

{
    name: 'dummy_teacher',
    username: 'dummy_teacher',
    email: 'dummy_teacher@gmail.com',
    role: 'teacher',
    password: 'qw'
}

{
    name: 'dummy_admin',
    username: 'dummy_admin',
    email: 'dummy_admin@gmail.com',
    role: 'admin',
    password: 'qw'
}
```



# Tech

* [Twitter Bootstrap]() - great UI boilerplate for modern web apps
* [Node.js]() - evented I/O for the backend
* [Express]() - fast node.js network app framework [@tjholowaychuk]
* [Mongodb](https://docs.mongodb.com/manual/introduction) - document based database

