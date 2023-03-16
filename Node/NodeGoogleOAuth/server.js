require('dotenv').config()

const USERS = require('./USERS')

const express = require('express')
const path = require('path')
const bcrypt = require('bcrypt')
const crypto = require('crypto')

const app = express()

/** JSON Parsing */
// parse application/x-www-form-urlencoded
app.use(express.urlencoded({ extended: true }))
// parse application/json
app.use(express.json())

const multer = require('multer')
app.use(multer().any())


const expressSession = require('express-session')

app.use(expressSession({
    secret: process.env.SESSION_SECRET,
    resave: false,
    saveUninitialized: false,
}))

const passport = require('passport')
const initializePassport = require('./auth/passport-config')
initializePassport(
    passport,
    email => USERS.find(u => u.email === email),
    id => USERS.find(u => u.id === id),
    (profile)=>{
        console.log(profile);
        return USERS[0];
    }
)
app.use(passport.initialize())
app.use(passport.session())

app.get('/', (req, res)=>{
    res.redirect('/login')
})
app.get('/login', (req, res)=>{
    res.sendFile(path.join(__dirname, 'html', 'login.html'))
})
app.get('/register', (req, res)=>{
    res.sendFile(path.join(__dirname, 'html', 'register.html'))
})

app.post('/login', passport.authenticate('local', {
    successRedirect: '/',
    failureRedirect: '/login',
}))

app.post('/register', async (req, res)=>{
    const user = {...req.body, id: crypto.randomUUID()}
    user.password = await bcrypt.hash(req.body.password, 10)
    USERS.push(user)
    res.json(USERS)
})

app.get('/auth/google', passport.authenticate('google', { scope: ['profile'] }));

app.get('/auth/google/callback', passport.authenticate('google', { failureRedirect: '/login' }),
    function(req, res) {
        // Successful authentication, redirect home.
        res.redirect('/');
    });

const checkAuthenticated = require('./middlewares/checkAuthenticated')

app.get('/protected', checkAuthenticated, (req, res)=>{
    console.log("user", req.user);
    res.json({ secret: 'secret' });
})

// Нельзя logout делать get запросом, но для примера норм
app.get('/logout', (req, res, next)=>{
    req.logout(function(err) {
        if (err) { return next(err); }
        res.redirect('/');
    });
})

app.use((err, req, res, next)=>{
    console.log(err)
    res.status(500).json({ error: err.message })
})

const PORT = process.env.PORT || 3000;
(async () => {
    try{
        app.listen(PORT, ()=>{
            console.log(`server is listening on port ${PORT}`);
        });
    }
    catch(e){
        console.log(e)
    }
})();