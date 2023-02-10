const express = require('express');
const app = express();


app.use((req, res, next)=>{
	console.log('always');
	next();
});

app.get('/a', (req, res)=>{
	console.log('/a has done');
	res.send({key: "value"});
});
app.get('/a', (req, res)=>{
	console.log('never');
});


app.get('/b', (req, res, next)=>{
	console.log('/b has not done yet');
	next();
});
app.use((req, res, next)=>{
	console.log('after /b always');
	next();
});
app.use('/b', (req, res, next)=>{
	console.log('b error generated');
	throw new Error('b did not done');
});
app.use('/b', (err, req, res, next)=>{
	console.log('error catched', err);
	next(err);
});


app.get('/c', (req, res)=>{
	console.log('c error generated');
	throw new Error('c did not done');
});
app.use('/c', (err, req, res, next)=>{
	console.log('c error founed');
	next();
});

app.use((err, req, res, next)=>{
	res.send('500');
});

app.use((req, res)=>{
	res.send('400');
});

const port = process.env.PORT || 3000;
app.listen(port, ()=>console.log(`port ${port}`)); 
