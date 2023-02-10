
const express = require('express');
const app = express();

const morgan = require('morgan');
const fs = require('fs');
switch(app.get('env')){
	case 'development':
		app.use(morgan('dev'));
		break;
	case 'production':
		const stream = fs.createWriteStream(__dirname + '/access.log', {flags: 'a'});
		app.use(morgan('combined', {stream}));
		break;
}

app.get('/', (req, res)=>{
	res.send({key: "value"});
});

const port = process.env.PORT || 3000;
app.listen(port, ()=>console.log(`port ${port}, env ${app.get('env')}`)); 
