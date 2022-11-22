const express = require('express');
const Router = express.Router();


Router.get('/', (req, res)=>{
	

	res.redirect(303, '/auth/login');
});

Router.use('/login', require('./login/login'));
Router.use('/register', require('./register/register'));
Router.use('/logout', require('./logout/logout'));


module.exports = Router;
