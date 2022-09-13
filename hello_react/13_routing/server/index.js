import express from 'express';
import path from 'path';
import fs from 'fs';

import React from 'react';
import ReactDOMServer from 'react-dom/server';

import {Home, About, Events, Contact, Products} from '../client/components/pages.js';
import Counter from '../client/components/Counter.js';

const app = express();

const PORT = process.env.PORT || 3000;

app.use(express.static(path.resolve(__dirname, '../build-client')));

app.get('/counter', (req, res)=>{
	res.sendFile(path.resolve(__dirname, '../build-client/index.html'));
});

// app.get('/', (req, res)=>{
	// res.send(path.resolve(__dirname, '../build-client/index.html'));
// });


app.listen(PORT, ()=>{
	console.log(`Server is listening on port ${PORT}`);
});