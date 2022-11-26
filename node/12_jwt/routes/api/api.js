const express = require('express');
const Router = express.Router();

const passport = require('passport');
Router.use(passport.authenticate('jwt', {session: false}));

Router.get('/', (req, res)=>{
	res.send({page: 'api'});
});

Router.use('/post', require('./post/post'));

module.exports = Router;
