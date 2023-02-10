const express = require('express');
const Router = express.Router();

Router.get('/', (req, res)=>{
	res.redirect(303, '/auth/login');
});

Router.use('/logout', require('./logout/logout'));

Router.use('/login', require('./login/login'));

Router.use('/register', require('./register/register'));

Router.use('/jwt', require('./jwt/jwt'));

module.exports = Router;
