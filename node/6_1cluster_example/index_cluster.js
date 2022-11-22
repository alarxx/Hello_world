const cluster = require('cluster');

const startServer = require('./index.js');

function startWorker(){
	const worker = cluster.fork();
	console.log(`Cluster: worker ${worker.id} started`);
}

if(cluster.isMaster){
	require('os').cpus().forEach(startWorker);

	cluster.on('disconnect', worker => console.log(`Cluster: worker ${worker.id} disconnected`));

	cluster.on('exit', (worker, code, signal)=>{
		console.log(`Cluster: worker ${worker.id} exit with code ${code} (${signal})`);
		startWorker();
	});
}
else {
	const port = process.env.PORT || 3000;
	startServer(port);
}