const { credentials } = require(`./config`);

const express = require('express');

const app = express();

const http = require('http');
const server = http.createServer(app);

/** Sessions */
/*const expressSession = require('express-session');
const expiresIn = 1000 * 60 * 60 * 24 * 7; // 7 days in milliseconds
const sessionMiddleware = expressSession({
    secret: credentials.cookieSecret,
    cookie: {
        maxAge: expiresIn,
    },
    // store: sessionStore, // connect-mongodb-session, например, или что похожая с Redis
    resave: false,
    saveUninitialized: false,
});
app.use(sessionMiddleware);*/


/** Доступ к статическим файлам */
app.use(express.static(__dirname + '/public'));

/** Body Parser */
const bodyParser = require('body-parser');
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

/** File Upload */
const fileUpload = require('express-fileupload');
app.use(fileUpload());

app.use(require('./routes/root'));

const port = process.env.PORT || 3000;
server.listen(port, ()=>{
    console.log(`Server is running on port ${port}`)
});