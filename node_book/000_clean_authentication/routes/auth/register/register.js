const express = require('express');
const passport = require('passport');

const register = require(require.main.path + '/auth/register');

const Router = express.Router();

Router.get('/', (req, res)=>{
	res.sendFile(require.main.path + '/views/auth/register.html');
});

Router.use('/', register('local'));

Router.post('/', passport.authenticate('local', {
	successRedirect: '/',
	failureRedirect: '/auth/register',
}));

Router.use('/', (err, req, res, next)=>{
	res.json({message: err});
});


module.exports = Router;
