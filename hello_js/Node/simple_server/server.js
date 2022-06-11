const http = require('http');
const port = 3000;
const server = http.createServer((req, res) => {
	const path = req.url.replace(/\/?(?:\?.*)?$/, '').toLowerCase();
	console.log(req.url);
	switch(path){
		case '':
			res.writeHead(200, {'Content-Type': 'text/plain'});
			res.end('NOthing');
			break;
		case '/about':
			res.writeHead(200, {'Content-Type': 'text/plain'});
			res.end('About');
			break;
		default: 
			res.writeHead(404, {'Content-Type': 'text/plain'});
			res.end('404');
			break;
	}
});

server.listen(port, () => console.log('listening on port: ${port}'));