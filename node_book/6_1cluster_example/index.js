function startServer(port){
	const express = require('express');
	const app = express();

	const cluster = require('cluster');

	app.use((req, res, next)=>{
		if(cluster.isWorker){
			console.log(`Worker ${cluster.worker.id} get request`);
		}
		next();
	});

	app.get('/', (req, res)=>{
		res.send({key: "value"});
	});

	app.listen(port, ()=>console.log(`port ${port}, env ${app.get('env')}`)); 
}

if (require.main === module) {
	const port = process.env.PORT || 3000;
	startServer(port);
}
else{	
	module.exports = startServer;
}