const express = require('express');
const passport = require('passport');


const Router = express.Router();

Router.get('/', (req, res)=>{
	res.sendFile(require.main.path + '/views/auth/register.html');
});

const register = require(require.main.path + '/auth/register');
Router.use('/', register('local'));

Router.use('/', passport.authenticate('local', {failureMessage: true, session: false}));

const {create} = require(require.main.path + '/auth/tokens');
Router.post('/', create());

module.exports = Router;
