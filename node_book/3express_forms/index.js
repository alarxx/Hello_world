/*
	Пробуем обработку форм с помощью HTML form и перехвата их с AJAX fetch
*/
const express = require('express');
const port = process.env.PORT || 3000;

const app = express();

const bodyParser = require('body-parser');
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());

app.use(express.static(__dirname + '/public'));


app.get('/', (req, res)=>{
	res.sendFile(__dirname + '/public/pages/index.html');
});
app.get('/img', (req, res)=>{
	res.sendFile(__dirname + '/public/pages/img.html');
});


//HMTL FORM
app.get('/newsletter/process', (req, res)=>{
	res.sendFile(__dirname + '/public/pages/html_form.html');
});
app.post('/newsletter/process', (req, res)=>{
	console.log(req.query);
	console.log(req.body);
	res.redirect('/');
});
//AJAX fetch
app.get('/signup', (req, res)=>{
	res.sendFile(__dirname + '/public/pages/ajax.html');
});
app.post('/api/signup', (req, res)=>{
	console.log(req.query);
	console.log(req.body);
	res.json({lol: "lolik"});
});


const multiparty = require('multiparty');

//MULTIPART FILE
app.get('/multipart', (req, res)=>{
	res.sendFile(__dirname + '/public/pages/multipart_form.html');
});
app.post('/multipart', (req, res)=>{
	const form = new multiparty.Form();
	form.parse(req, (err, fields, files) => {
		if(err){ 
			res.status(500).send({error: err.message});
		} 
		else {
			console.log("fields: ", fields);
			console.log("files: ", files);
			res.redirect(303, '/multipart');
		}
	});
});
//MULTIPART FILE AJAX
app.get('/api/multipart/', (req, res)=>{
	res.sendFile(__dirname + '/public/pages/multipart_form_ajax.html');
});
app.post('/api/multipart/:number', (req, res)=>{
	console.log(req.query);
	console.log(req.params);
	console.log(req.body);
	const form = new multiparty.Form();
	form.parse(req, (err, fields, files) => {
		if(err){ 
			res.status(500).json({error: err.message});
		} 
		else {
			console.log("fields: ", fields);
			console.log("files: ", files);
			res.json({result: 'success'});
		}
	});
});


app.listen(port, ()=>console.log(`server listening on port ${port}`));

 
