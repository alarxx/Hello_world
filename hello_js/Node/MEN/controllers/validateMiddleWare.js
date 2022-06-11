module.exports = (req, res, next)=>{
	if(req.files == null || req.body == null || req.body){
		return res.redirect('/posts/new');
	}
	next();
};