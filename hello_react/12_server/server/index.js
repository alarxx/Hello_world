import express from 'express';
import path from 'path';
import fs from 'fs';

import React from 'react';
import ReactDOM from 'react-dom/server';
import Main from '../src/components/Main.js';
import About from '../src/components/About.js';

const app = express();

const PORT = process.env.PORT || 3000;

app.use(express.static(path.resolve(__dirname, '../dist')));

app.get('/*', (req, res)=>{
	const app = ReactDOM.renderToString(
		<Main />
	);

	const indexFile = 'index.html';

	fs.readFile(indexFile, 'utf8', (err, data)=>{
		return res.send(
			data
		);
	});

});

app.listen(PORT, ()=>{
	console.log(`Server is listening on port ${PORT}`);
});