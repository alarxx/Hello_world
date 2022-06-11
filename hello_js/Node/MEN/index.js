const express = require('express');
const path = require('path');
const ejs = require('ejs');
const mongoose = require('mongoose');
const bodyParser = require('body-parser');
const fileUpload = require('express-fileupload');

mongoose.connect('mongodb://localhost/my_database', {useNewUrlParser: true});

const app = express();
app.set('view engine', 'ejs');

const customMiddleWare = require('./controllers/customMiddleWare');
const validateMiddleWare = require('./controllers/validateMiddleWare');

app.use(express.static('public'));

app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json());

app.use(fileUpload());

//app.use(customMiddleWare);

const storeController = require('./controllers/store');
app.post('/posts/store', storeController);

app.listen(3000, ()=>console.log(3000));

const homeController = require('./controllers/home');
app.get('/', homeController);

app.get('/home', homeController);

const aboutController = require('./controllers/about');
app.get('/about', aboutController);

const postController = require('./controllers/post');
app.get('/post', (req, res) => {
	res.render('post');
});

const contactController = require('./controllers/contact');
app.get('/contact', contactController);

const newPostController = require('./controllers/newPost.js');
app.get('/posts/new', newPostController);

const newUserController = require('./controllers/newUser.js');
app.get('/auth/register', newUserController);

const storeUserController = require('./controllers/storeUser.js');
app.get('/users/register', newUserController);

app.get('*', (req, res)=>
	res.status(404).send("404")
);
