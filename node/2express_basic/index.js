const express = require('express');
const port = process.env.PORT || 3000;

const app = express();

app.get('/', (req, res)=>{
	// res.status(200); // default status code
	res.type('text/plain');
	res.send('[Home page]');
});

app.get('/about', (req, res)=>{
	res.type('text/plain');
	res.send('[About page]');
});

app.get('/about/contact', (req, res)=>{
	res.type('text/plain');
	res.send('[About Contact page]');
});

app.use((req, res)=>{
	res.type('text/plain');
	res.status(404);
	res.send('404 - Not Found');
});

app.use((err, req, res, next) => {
	console.error(err.message);
	res.type('text/plain');
	res.status(500);
	res.send('500 - Server Error');
});

app.listen(port, ()=>console.log(`server listening on port ${port}`));