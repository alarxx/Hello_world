const http = require('http');
const fs = require('fs');
const port = process.env.PORT || 3000;

function serveStaticFile(res, path, contentType, responseCode=200){
	fs.readFile(__dirname + path, (err, data)=>{
		if(err){
			res.writeHead(500, {'Content-type': 'text/plain'});
			return res.end('500 - Внутренняя ошибка');
		}
		res.writeHead(responseCode, {'Content-type': contentType});
		res.end(data);
	});
}

const server = http.createServer((req, res)=>{
	console.log(req.url);

	/**
	*	Приводим URL к единому виду, удаляя
	*	строку запроса , необязательную косую черту
	*	в конце с троки и переводя в нижний регистр,
	* 	Почему-то без regexp тоже работает
	*/
	const url_path = req.url.replace(/\/?(?:\?.*)?$/, '').toLowerCase();
	
	switch(url_path){
		case '':
			// res.writeHead(200, {'Content-type': 'text/plain'});
			// res.end('Home Page');
			serveStaticFile(res, '/public/pages/index.html', 'text/html');
			break;
		case '/about':
			// res.writeHead(200, {'Content-type': 'text/plain'});
			// res.end('About Page');
			serveStaticFile(res, '/public/pages/about.html', 'text/html');
			break;
		case '/img/logo.jpg':
			serveStaticFile(res, '/public/img/logo.jpg', 'image/jpeg');
			break;
		default:
			// res.writeHead(404, {'Content-type': 'text/plain'});
			// res.end('Not Found');
			serveStaticFile(res, '/public/pages/404.html', 'text/html');
			break;
	}
}); 

server.listen(port, () => console.log(`сервер запущен на порте ${port}`));
