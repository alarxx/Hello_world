const express = require('express');
const Router = express.Router();

Router.get('/', (req, res)=>{
	res.send({page: 'home', name: req.user?.name});
});

Router.use('/auth', require('./auth/auth'));
Router.use('/api', require('./api/api'));

// error handling
Router.use((err, req, res, next)=>{
	res.json({
		status: 'fail',
		message: err
	});
});

module.exports = Router;
