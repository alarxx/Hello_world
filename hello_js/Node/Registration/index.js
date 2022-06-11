const express = require('express');
const mongoose = require('mongoose');
const ejs = require('ejs');
const bodyParser = require('body-parser');
const expressSession = require('express-session');
const User = require('./models/User.js');

const app = new express();

mongoose.connect('mongodb://localhost/users_db', {useNewUrlParser: true});

app.use(express.static('public'));
app.set('view engine', 'ejs');

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({extended:true}));

app.use(expressSession({
	secret: 'keyboard cat'
}));

app.get('/', async (req, res) => {
	const users = await User.find({});
	console.log(req.session);
	res.render('index', {users})
});

//const authMiddleWare = require('./middlewares/authMiddleWare.js');
const registerController = require('./controllers/register');
app.get('/auth/register', registerController);

const storeUserController = require('./controllers/storeUser.js');
app.post('/users/register', storeUserController);

app.get('/auth/login', (req, res)=>{
	res.render('login');
});
const loginController = require('./controllers/login.js');
app.post('/users/login', loginController);

const port = 4000;

app.listen(port, () => {
	console.log(`listening on ${port}`);
});
