const mongoose = require('mongoose');
const BlogPost = require('./models/BlogPost');
mongoose.connect('mongodb://localhost/my_database', {useNewUrlParser: true});

/*
BlogPost.create({
	title: 'Title', 
	body: 'Body AAAAAAAAAAAAAAAAAAAAAAAAA'
}, (error, blogpost) => {
	console.log(error,blogpost);
});
*/

let arr = BlogPost.find({title: "Title"}, (error, blogpost)=>{});
console.log(arr._id);