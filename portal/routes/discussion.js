const axios = require('axios');
var express = require("express");
var schedule = require('node-schedule');
var router = express.Router();
var Discussion = require("../model/discussion");
var Answers = require("../model/answers");
var Comments = require("../model/comments");
var Resource = require("../model/resource");
var Pathway = require("../model/learningPathways");
var LearningMapResource = require("../model/learningMapResources");
var LearningMap = require("../model/learningMap");

var threshold = 120*1000; //5 days = 5*24*60*60*1000

var Authenticator = require("../middleware/authenticator");
var ObjectId = require('mongoose').Types.ObjectId;
// Get all paths/resources queried map
router.get("/m/:id", Authenticator.ensureAuthenticated, async (req, res) => {
    const m_id = req.params.id;
    const map = await LearningMap.getLearningMapById(m_id);
    const discussion = await Discussion.find({});
    Discussion.find().distinct('res_id', async function(error, ids) {
        var resource = ids;
        var resourceNames = [];
        var i;
        var m_resource = [];
        for(i=0;i<resource.length;i++){
            if(ObjectId.isValid(resource[i])){
                var r = await Resource.getResourceByResourceId(resource[i]);
                const mapdata = await LearningMapResource.find({resourceId : r._id});
                if(mapdata.length!=0){
                    if(m_id == mapdata[0].mapId){   
                        m_resource.push(resource[i]);
                        resourceNames.push(r.resourceName);
                    }
                }
            }
            else{
                const check = await Pathway.find({pathwayId : resource[i]});
                if(m_id == check[0].mapId){
                    m_resource.push(resource[i]);
                    resourceNames.push("");
                }
                
            }
        }
        res.render("alldiscussion",{
            layout: "layoutDiscussions",
            user: req.user,
            username: req.user.username,
            discussions: discussion,
            r_names: resourceNames,
            r_ids: m_resource,
            mapName: map[0].mapName
        });
    }); 
});
// Get all discussions page for the queried resource type discussion
router.get("/r/:id", Authenticator.ensureAuthenticated, async (req, res) => {
    const discussion = await Discussion.find({res_id : req.params.id});
    if(ObjectId.isValid(req.params.id)){
        const check = await Resource.getResourceByResourceId(req.params.id);
        const mapdata = await LearningMapResource.find({resourceId : check._id});
        if(mapdata.length == 0){
            m_id = 'none';
        }
        else{
            m_id = mapdata[0].mapId;
        }
        if(check != ""){
            res.render("Discussions",{
                layout: "layoutDiscussions",
                user: req.user,
                username: req.user.username,
                discussions: discussion,
                r_id: req.params.id,
                res: check,
                dtype: "resource",
                m_id: m_id
              });
        }
        else{
            res.send("not found")
        }
    }
    else{
        res.send("not found")
    }
});
// Get all discussions page for the queried path type discussion
router.get("/p/:id", Authenticator.ensureAuthenticated, async (req, res) => {
    const discussion = await Discussion.find({res_id : req.params.id});
    const check = await Pathway.find({pathwayId : req.params.id});
    if(check != ""){
        res.render("Discussions",{
            layout: "layoutDiscussions",
            user: req.user,
            username: req.user.username,
            discussions: discussion,
            r_id: req.params.id,
            dtype: "path",
            m_id: check[0].mapId
        });
    }
    else{
        res.send("not found")
    }
    
});
// post the question for the given resource type discussion
router.post('/r/:id', Authenticator.ensureAuthenticated, async (req, res) => {
    var discussion = new Discussion({dtype: "resource",res_id: req.params.id, title: req.body.title,question: req.body.question,username: req.user.username, userid: req.user._id, timestamp: Date.now()});
    await discussion.save();
    res.redirect('/discussions/r/'+req.params.id);
})
// post the question for the given path type discussion
router.post('/p/:id', Authenticator.ensureAuthenticated, async (req, res) => {
    var discussion = new Discussion({dtype: "path",res_id: req.params.id, title: req.body.title,question: req.body.question,username: req.user.username, userid: req.user._id, timestamp: Date.now()});
    await discussion.save();
    res.redirect('/discussions/p/'+req.params.id);
})

// Get start discussion page for a resource type discussion
router.get("/start/r/:id", Authenticator.ensureAuthenticated, async (req, res) => {
    res.render("startDiscussion",{
        layout: "layoutDiscussions",
        user: req.user,
        username: req.user.username,
        r_id: req.params.id,
        dtype: "resource"
      });
});
// Get start discussion page for a path type discussion
router.get("/start/p/:id", Authenticator.ensureAuthenticated, async (req, res) => {
    res.render("startDiscussion",{
        layout: "layoutDiscussions",
        user: req.user,
        username: req.user.username,
        r_id: req.params.id,
        dtype: "path"
      });
});
// upvote for the question
router.post('/question/:id', Authenticator.ensureAuthenticated, async (req, res) => {
    var val = req.body.value;
    if (Number(val) === 1){
        await Discussion.findByIdAndUpdate(req.params.id, { "$addToSet": { "votes": req.user._id } }, { "new": true, "upsert": true })
    }
    else{
        await Discussion.findByIdAndUpdate(req.params.id, { "$pull": { "votes": req.user._id } }, { "new": true, "upsert": true })
    }
    var d = await Discussion.findById(req.params.id).populate('answers');
    res.send({discussion: d});
})
// upvote for the answer
router.post('/answer/:id', Authenticator.ensureAuthenticated, async (req, res) => {
    var val = req.body.value;
    if (Number(val) === 1){
        await Answers.findByIdAndUpdate(req.params.id, { "$addToSet": { "votes": req.user._id } }, { "new": true, "upsert": true })
    }
    else{
        await Answers.findByIdAndUpdate(req.params.id, { "$pull": { "votes": req.user._id } }, { "new": true, "upsert": true })
    }
    var a = await Answers.findById(req.params.id);
    var d = await Discussion.findById(req.body.did).populate('answers');
    res.send({discussion: d, answer: a});
})
// TA verification for the answer
router.post('/a/verify/:id', Authenticator.ensureAuthenticated, async (req, res) => {
    var val = req.body.value;
    if (Number(val) === 1){
        await Answers.findByIdAndUpdate(req.params.id, { TA_verified: true});
    }
    else{
        await Answers.findByIdAndUpdate(req.params.id, { TA_verified: false});
    }
    var d = await Discussion.findById(req.body.did).populate('answers');
    var a = await Answers.findById(req.params.id);
    res.send({discussion: d, answer: a});
})
// upvote for the comment
router.post('/comment/:id', Authenticator.ensureAuthenticated, async (req, res) => {
    var val = req.body.value;
    if (Number(val) === 1){
        await Comments.findByIdAndUpdate(req.params.id, { "$addToSet": { "votes": req.user._id } }, { "new": true, "upsert": true })
    }
    else{
        await Comments.findByIdAndUpdate(req.params.id, { "$pull": { "votes": req.user._id } }, { "new": true, "upsert": true })
    }
    var c = await Comments.findById(req.params.id);
    var d = await Discussion.findById(req.body.did).populate({
        path: 'answers',
        populate: { path: 'comments' }
      });
    res.send({discussion: d, comment: c});
})
// TA verification for the comment
router.post('/c/verify/:id', Authenticator.ensureAuthenticated, async (req, res) => {
    var val = req.body.value;
    if (Number(val) === 1){
        await Comments.findByIdAndUpdate(req.params.id, { TA_verified: true});
    }
    else{
        await Comments.findByIdAndUpdate(req.params.id, { TA_verified: false});
    }
    var d = await Discussion.findById(req.body.did).populate({
        path: 'answers',
        populate: { path: 'comments' }
      });
      var c = await Comments.findById(req.params.id);
    res.send({discussion: d, comment: c});
})
// Annotate thread till comment(Used for retraining the model)
router.post('/annotate/:id/:cid', Authenticator.ensureAuthenticated, async (req, res) => {
    await Comments.findByIdAndUpdate(req.params.cid, { annotate: req.body.annotate});
    res.redirect(`/discussions/${req.params.id}`);
})
// Close thread
router.post('/branch/:id/:aid', Authenticator.ensureAuthenticated, async (req, res) => {
    await Answers.findByIdAndUpdate(req.params.aid, { thread_close: true});
    res.redirect(`/discussions/${req.params.id}`);
})
// Posting the answer
router.post('/:id/answer', Authenticator.ensureAuthenticated, async (req, res) => {
    const discussion = await Discussion.findById(req.params.id);
    const answer = new Answers({answer: req.body.answer,username: req.user.username, userid: req.user._id, timestamp: Date.now()});
    discussion.answers.push(answer);
    await answer.save();
    await discussion.save();
    res.redirect(`/discussions/${discussion._id}`);
});
// Posting the comment
router.post('/:id/:aid/comment', Authenticator.ensureAuthenticated, async (req, res) => {
    const discussion = await Discussion.findById(req.params.id).populate({
        path: 'answers',
        populate: { path: 'comments' }
      });
    const answer = await Answers.findById(req.params.aid);
    const comment = new Comments({comment: req.body.comment,username: req.user.username, userid: req.user._id, timestamp: Date.now()});
    answer.comments.push(comment);
    // save comment and load discussion again so that new comment is sent while api call
    await comment.save();
    await answer.save();
    await discussion.save();
    const discussion2 = await Discussion.findById(req.params.id).populate({
        path: 'answers',
        populate: { path: 'comments' }
      });
    const answer2 = await Answers.findById(req.params.aid);
    answer2.inactive = false;
    var index = 0;
    for (var i = 0;i<discussion2.answers.length;i++){
        if(discussion2.answers[i]._id.equals(answer2._id)){
            index = i;
        }
    }
    data = {
        discussion: discussion2, 
        index: index
    }
    var date = new Date(comment.timestamp+threshold);
    var j = schedule.scheduleJob(date, async function(date,a=answer2.comments.length,b=data){
            var answer3 = await Answers.findById(req.params.aid);
            console.log(a);
            console.log(answer3.comments.length);
            if(answer3.comments.length == a){
                await axios({
                    method: 'post',
                    url: 'http://127.0.0.1:8000/disc/branch',
                    data: data
                }).then(async function (response) {
                    if(response.data == 2) answer3.Branch_result = 'Agreement';
                    if(response.data == 1) answer3.Branch_result = 'Partial';
                    if(response.data == 0) answer3.Branch_result = 'Disagreement';
                    answer3.inactive = true;
                    await answer3.save()
                    var msg = "TA Intervention needed ";
                    msg += String(req.params.id);
                    await axios({
                        method: 'post',
                        url: 'http://127.0.0.1:5555/resources/addTAnotification',
                        data: {message:msg}
                    }).then(async function (response) {
                        console.log("Added");
                    })
                    // console.log(answer3.Branch_result);
                    // console.log(answer3.no_intervene_requested);
                })
                .catch(function (error) {
                    console.log(error);
                });
            }
    });
    if( req.user.role === "user"){
        await axios({
            method: 'post',
            url: 'http://127.0.0.1:8000/disc/result',
            data: data
        }).then(function (response) {
            // console.log(response.data);
            if(response.data == 1){
                answer2.TA_intervene = true;
                answer2.no_intervene_requested += 1;
            }
        })
        .catch(function (error) {
            console.log(error);
        });
    }
    else{
        answer2.TA_intervene = false;
    }
    await comment.save();
    await answer2.save();
    await discussion2.save();
    
    res.redirect(`/discussions/${discussion._id}`);
});
// Get the discussion thread
router.get('/:id', Authenticator.ensureAuthenticated, async (req, res,) => {
    const discussion = await Discussion.findById(req.params.id).populate({
        path: 'answers',
        populate: { path: 'comments' }
      });
    res.render("Discuss", { layout: "layoutDiscussions",
    user: req.user,
    username: req.user.username,
    discussion: discussion });
});

module.exports = router;
