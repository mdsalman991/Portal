# Component Title : Navigated Learning Portal
Our work majorly capitalizes on developing a web portal which can provide an interface to access the resources in a pathway that is user specific.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Installation

#Linux Based Systems:
Step 1) Update your system:
sudo apt-get update

Step 2) Install Node.js:
sudo apt-get install nodejs 

Step 3) Install NPM:
sudo apt-get install npm 

nodejs -v 

Step 4) Now install the given libraries:
> npm install express
> npm install express-session
> npm install mysql
> npm install body-parser
> npm install formidable
> npm install cors
> npm install nodemon
> npm install multer
> npm install uuid

Step 5) Install mysql in your system:
sql ubuntu 
sudo apt install mysql-server
sudo mysql_secure_installation

#Windows Based Systems:

Step 1) To download the suitable .msi file according to your system's configuration. Visit this link
https://nodejs.org/en/download/ 

Step 2) Run the downloaded Node.js .msi Installer - including accepting the license, selecting the destination, and authenticating for the install.

This requires Administrator privileges, and you may need to authenticate

To ensure Node.js has been installed, run node -v in your terminal - you should get something like v6.9.5
Update your version of npm with npm install npm --global

This requires Administrator privileges, and you may need to authenticate 
npm install npm --global 

Step 3) Install mysql in your system:
Download the MySQL Installer from dev.mysql.com. 

## Usage 
1)  Run wsl.sql in your mysql terminal.
	 mysql> source <completepath>/wsl.sql
2)  Open your terminal and navigate to the directory where server2.js file is present.
	update username and password for your sql database in server2.js file.
3)  Run server2.js in terminal.
	 node server2.js
4)  In chrome or any browser type localhost:5555 
5)  Login using your gmail account.
6)  You will be directed to the signup page,
         enter your details there to login as 'user' or 'teacher' respectively.
7)  On the user's dashboard, you have various options to access the map and upload links.
8)  On the teacher's dashboard, you have options to upload resources from your local system 	and create new maps.
9)  You can edit your profile using the drop-down menu, 
         by clicking on 'edit profile'.
10) You can sign out from the portal by clicking on the log out button.

## Parameters and Performance 
1) Open terminal in the project directory and run the following command:
   $ node server2.js
2) You can see the output on any browser at localhost:5555/.
3) The response time of the portal depends on the internet bandwidth.

## REST API 
Everyone is required to mandatorily expose the below-mentioned endpoints - 
* /notify - Element of Message Passing / Notification System
* /receive - Element of Message Passing / Notification System
 
## GMAIL API
We have implemented gmail API for login and logout mechanism.

## Running the tests
Explain how to run the automated tests for this component

Login to the portal:
1)As student: 
	On the student's dashboard the user can access the resources, view the maps, upload links, edit profile and view profile 
2)As teacher:
	On the teacher's dashboard the teacher can create a new map, add the resources from the local system, edit profile and view profile

## Authors 

1. Ankita Bisht
2. Archit Semwal
3. Shashank Agarwal

## Project Status
Dockerization of the portal on college server.


