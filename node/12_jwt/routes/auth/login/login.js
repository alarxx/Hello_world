const express = require('express');
const passport = require('passport');


const Router = express.Router();


Router.get('/', (req, res)=>{
	res.sendFile(require.main.path + '/views/auth/login.html');
});

Router.use('/', passport.authenticate('local', {failureMessage: true, session: false}));

const {create} = require(require.main.path + '/auth/tokens');
Router.post('/', create());

module.exports = Router;
