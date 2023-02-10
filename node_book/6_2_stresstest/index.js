const express = require('express');
const app = express();

app.get('/', (req, res)=>{
	res.send({key: "value"});
});

const port = process.env.PORT || 3000;
app.listen(port, ()=>console.log(`port ${port}, env ${app.get('env')}`)); 
