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
const {LocalStrategy} = require('./auth/passport');
LocalStrategy();


app.use('/', require('./routes/root'));


const port = process.env.PORT || 3000;
app.listen(port, ()=>console.log(`port ${port}`));
