const express = require('express');
const passport = require('passport');


const Router = express.Router();


Router.get('/', (req, res)=>{
	res.sendFile(require.main.path + '/views/auth/login.html');
});

Router.post('/',
	passport.authenticate('local', {
		successRedirect: '/',
		failureRedirect: '/auth/login',
	})
);


module.exports = Router;
