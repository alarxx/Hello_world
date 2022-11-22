const {credentials} = require(`./config`);


const mongoose = require('mongoose');
mongoose.connect(credentials.dbUri, {useNewUrlParser: true});


const express = require('express');
const app = express();


app.use(express.static(__dirname + '/public'));


const bodyParser = require('body-parser');
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({extended: true}));

const fileUpload = require('express-fileupload');
app.use(fileUpload());


const expressSession = require('express-session');
app.use(expressSession({
	resave: false,
	saveUninitialized: false,
	secret: credentials.cookieSecret
}));


const passport = require('passport');
app.use(passport.initialize());
app.use(passport.session());
require('./auth/passport');


const checkAuthenticated = (req, res, next) => {
	if(req.isAuthenticated())
		next();
	else
		res.redirect(303, '/login');
};
const checkNotAuthenticated = (req, res, next) => {
	if(req.isAuthenticated())
		res.redirect(303, '/login');
	else
		next();
};

app.get('/', (req, res)=>{
	console.log(req.user);
	res.send({data: 'home page'});
});


app.get('/protected', checkAuthenticated, (req, res)=>{
	res.send({data: "secret"});
});

const UserModel = require('./models/User');
app.get('/error', (req, res)=>{
	const new_user = {};
	UserModel.create(new_user, (err, user)=>{
		if(err){
			const errors = Object.keys(err.errors).map(key => err.errors[key].message);
			console.log(errors);
			return res.redirect('/login');
		}
		res.redirect('/');
	});
});

app.get('/login', (req, res)=>{
	res.sendFile(__dirname + '/views/login.html');
});

app.post('/login', 
	passport.authenticate('local', {
		successRedirect: '/',
		failureRedirect: '/login',
	})
);

app.get('/logout', (req, res)=>{
	req.logOut(()=>{});
	res.redirect(303, '/');
});

const port = process.env.PORT || 3000;
app.listen(port, ()=>console.log(`port ${port}`)); 
