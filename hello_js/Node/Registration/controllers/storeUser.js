const User = require('../models/User.js');
const path = require('path');

module.exports = async (req, res)=>{
	//console.log(req.body);
	await User.create(req.body, (error, user)=>{
		console.log(error);
		if(error){
			const validationErrors = Object.keys(error.errors).map(key => error.errors[key].message);
			req.session.validationErrors = validationErrors;
			res.redirect('/auth/register');
		}
		else 
			res.redirect('/');
	});
}