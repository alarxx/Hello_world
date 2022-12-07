(async () => {
    const express = require('express');

    const mongoose = require('mongoose');
    const dbUri = 'mongodb://127.0.0.1:27017/mongo';
    await mongoose.connect(dbUri, {useNewUrlParser: true});

    const app = express();

    const bodyParser = require('body-parser');
    app.use(bodyParser.json());
    app.use(bodyParser.urlencoded({extended: true}));

    const fileUpload = require('express-fileupload');
    app.use(fileUpload());

    const expiresIn = 1000 * 60 * 60 * 24 * 7; // 7 days in milliseconds
    const session = require('express-session');
    const MongoDBStore = require('express-mongodb-session')(session);
    const store = new MongoDBStore({
        uri:  dbUri,
        collection: 'sessions',
        expires: expiresIn,
    });
    store.on('error', function(error) {
        console.log(error);
    });

    app.use(session({
        secret: 'secret cat',
        cookie: {
            maxAge: expiresIn,
            sameSite: 'strict',
        },
        store: store,
        resave: false,
        saveUninitialized: false,

    }));

    const passport = require('passport');
    app.use(passport.initialize());
    app.use(passport.session());
    const LocalStrategy = require('./auth/passport');
    LocalStrategy();

    app.use('/', require('./routes/root.js'));

    const port = process.env.PORT || 3000;
    app.listen(port, ()=>console.log(`listening on port ${port}`))

})();
