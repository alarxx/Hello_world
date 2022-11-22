const {credentials} = require(`./config`);

const express = require('express');
const fileUpload = require('express-fileupload');
const expressSession = require('express-session');
const mongoose = require('mongoose');

mongoose.connect(credentials.dbUri, {useNewUrlParser: true});

const app = express();

app.use(express.static(__dirname + '/public'));

app.use(fileUpload());

app.use(expressSession({
	resave: true,
	saveUninitialized: true,
	secret: credentials.cookieSecret
}));

const PostSchema = new mongoose.Schema({
	title: String,
	body: String
});
const Post = mongoose.model('Post', PostSchema);

app.get('/', (req, res)=>{
	//CREATE
	// Post.create({
	// 	title: "Title",
	// 	body: "Body"
	// }, (err, post)=>{
	// 	console.log(err, post);
	// });

	//FIND
	// Post.find({}, (err, posts)=>{
	// 	console.log(err, posts);
	// });
	res.send({key: 'value'});
});


const port = process.env.PORT || 3000;
app.listen(port, ()=>console.log(`port ${port}`)); 
