const express = require('express');
const path = require('path');
const fs = require('fs');
const multiparty = require('multiparty');
const fileUpload = require('express-fileupload');

const app = express();

// app.use(fileUpload());

app.get('/api/local', (req, res)=>{
	res.sendFile(__dirname + '/views/home.html');
});
app.post('/api/local', (req, res)=>{
	const form = new multiparty.Form({autoFiles: true});
	form.parse(req, (err, fields, files) => {
		if(err){ 
			return res.status(500).json({error: err.message});
		} 
		else {
			console.log("fields: ", fields);
			console.log("files: ", files.photo[0]);
		}
	});

	// const image = req.files.photo;
	// console.log(image);
	res.redirect(303, '/api/local');

});



const port = process.env.PORT || 3000;
app.listen(port, ()=>console.log(`port ${port}`)); 
