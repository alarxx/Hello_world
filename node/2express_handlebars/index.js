const express = require('express');
const expressHandlebars = require('express-handlebars');
const port = process.env.PORT || 3000;

const app = express();

/* HANDLEBARS, like EJS*/
app.engine('handlebars', expressHandlebars.engine({
	defaultLayout: 'main'
}));
app.set('view engine', 'handlebars');
app.set('views', './views');


/* 
* Static Content 
* Создает маршрут на каждый файл в папке /public
*/
app.use(express.static(__dirname + '/public'));

/* Dynamic Content */
// const fortunes = ["one","two","three","four","five", "six"];

app.get('/', (req, res)=>{
	const fortune = require('./components/fortune');
	console.log(fortune);
	// const randomFortune = fortunes[Math.floor(Math.random() * fortunes.length)];
	res.render('home', {fortune: fortune()});
});

app.get('/about', (req, res)=>{
	res.render('about');
});

app.get('/headers', (req, res)=>{
	console.log(req.url);
	res.type('text/plain');
	// Деструктуризация массива
	// const [one, two, three, four] = [1, 2, 3, 4]; 
	const headers = Object.entries(req.headers)
			.map(([key, value])=>`${key} ${value}`)
			.join('\n');
	res.send(headers);
});

app.use((req, res)=>{
	res.status(404);
	res.render('404');
});

app.use((err, req, res, next) => {
	console.error(err.message);
	res.status(500);
	res.render('500');
});

app.listen(port, ()=>console.log(`server listening on port ${port}`));