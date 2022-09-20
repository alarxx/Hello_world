/*
	req.url

	res.writeHead(status, contentType);
	res.end(data);

	Статические ресурсы раздаются точно так же как и обычная маршрутизация
*/

const http = require('http');
const fs = require('fs');
const port = process.env.PORT || 3000;

function serverStaticFile(res, path, contentType, responseCode=200){
	fs.readFile(__dirname + path, (err, data) => {
		// console.log(err, data);
		if(err){
			res.writeHead(500, {'Content-Type': 'text/plain'});
			return res.end('500 - ERROR');
		}
		res.writeHead(responseCode, {'Content-Type': contentType});
		res.end(data);
	});
}

const server = http.createServer((req, res)=>{
	// Сам req json очень большой
	console.log(req.url);
	
	const url_path = req.url.replace(/\/?(?:\?.*)?$/, '').toLowerCase();

	switch(url_path){
		case '':
			serverStaticFile(res, '/public/pages/home.html', 'text/html');
			//res.writeHead(200, {'Content-Type': 'text/plain'});
			//res.end('[Home Page]');
			break;

		case '/about':
			serverStaticFile(res, '/public/pages/about.html', 'text/html');
			//res.writeHead(200, {'Content-Type': 'text/plain'});
			//res.end('[About Page]');
			break;

		case '/images/lol.jpg':
			serverStaticFile(res, '/public/images/lol.jpg', 'image/jpeg');
			break;

		default: 
			serverStaticFile(res, '/public/pages/404.html', 'text/html');
			//res.writeHead(404, {'Content-Type': 'text/plain'});
			//res.end('Not Found');
			break;
	}
});

server.listen(port, ()=>console.log(`server running on port ${port}`));