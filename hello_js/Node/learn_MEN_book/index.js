const express = require('express');
const ejs = require('ejs');
const bodyParser = require('body-parser');

const app = express();
const port = 3000;

app.use(express.static('public'));
app.set('view engine', 'ejs');

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({extended:true}));

app.get('/', (req, res)=>{
	res.render('index');
});

app.get('/home', (req, res)=>{
	res.render('index');
});

app.get('/about', (req, res)=>{
	res.render('about');
});

app.get('/post', (req, res)=>{
	res.render('post');
});

app.get('/contact', (req, res)=>{
	res.render('contact');
});

app.listen(port, ()=>{
	console.log(`listening on port ${port}`);
});