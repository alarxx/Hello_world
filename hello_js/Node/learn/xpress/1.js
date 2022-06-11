const express = require('express');
const path = require('path');
const port = 3000;

const app = express();

app.use(express.static('public'));

app.listen(port, ()=>console.log(`Server listening on port: ${port}`));

app.get('/', (req, res) => {
	res.json({name: 'Greg'});
});

app.get('/home', (req, res) => {
	res.sendFile(path.resolve(__dirname, 'index.html'));
});