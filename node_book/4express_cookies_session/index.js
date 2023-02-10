/*
Сессии базируются на кукисах. Сессии удобнее, чем куки.
Я пока не вижу большого смысла в кукисах и сессиях, 
кроме как хранить в них unique id для сессии.

Cookie можно сделать безопасными и небезопасными.
Их можно кодировать на стороне сервера, 
можно изменять только сервером, передавать только через https.

Использовать сессии удобнее можно не задумываться ни о чем, 
просто добавлять филды в req.session, удалять, когда надо.
Для cookie напрямую нужно указывать подписан или нет, все детали.

Можно не использовать express-session, а просто написать свой 
middleware-обработчик req.cookies, res.cookie и так далее.

*/


const express = require('express');

const port = process.env.PORT || 3000;

const app = express();


app.use(express.static(__dirname + '/public'));


const bodyParser = require('body-parser');
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());


const multiparty = require('multiparty');


const {credentials} = require('./config');

const cookieParser = require('cookie-parser');
app.use(cookieParser(credentials.cookieSecret));


app.get('/', require('./routes/handlers').home);

app.get('/cookies', (req, res)=>{
	res.cookie('monster', 'namnam');
	res.cookie('signed_monster', 'signed namnam', {signed: true});
	res.redirect('/');
});

app.get('/check', (req, res)=>{
	console.log(req.cookies.monster);
	console.log(req.signedCookies.signed_monster);
	res.redirect('/');
});

app.get('/clear', (req, res)=>{
	res.clearCookie('monster');
	res.clearCookie('signed_monster');
	res.redirect('/')
});


const expressSession = require('express-session');
app.use(expressSession({
	resave: true,
	saveUninitialized: true,
	secret: credentials.cookieSecret
}));

app.get('/session', (req, res)=>{
	console.log(req.session);
	res.json({session: req.session.userName});
});
app.get('/setsession', (req, res)=>{
	req.session.userName = "Alar123";
	res.redirect('/session');
});
app.get('/delsession', (req, res)=>{
	delete req.session.userName;
	res.redirect('/session');
});

app.listen(port, ()=>console.log(`server is listening on port ${port}`));