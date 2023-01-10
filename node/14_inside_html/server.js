const express = require('express');
const path = require('path');

const app = express();

app.use(express.static(path.resolve(__dirname, 'view')));

app.use('/socket.io-client', express.static(path.resolve(__dirname, '/node_modules/socket.io-client/dist/')));

const port = process.env.PORT || 3000;
app.listen(port, ()=>console.log(`port ${port}`));