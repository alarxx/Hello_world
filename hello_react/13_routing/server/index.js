import express from 'express';
import path from 'path';
import fs from 'fs';

import React from 'react';
import ReactDOMServer from 'react-dom/server';

const app = express();

const PORT = process.env.PORT || 3000;

app.use(express.static(path.resolve(__dirname, '../build-client')));

app.get('/about', (req, res)=>{
	res.sendFile(path.resolve(__dirname, '../build-client/index.html'));
});

app.get('*', (req, res)=>{
	res.sendFile(path.resolve(__dirname, '../build-client/index.html'));
});

app.listen(PORT, ()=>{
	console.log(`Server is listening on port ${PORT}`);
});