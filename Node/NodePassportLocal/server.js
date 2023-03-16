require('dotenv').config()

const express = require('express');
const path = require('path');

const app = express();

/** JSON Parsing */
// parse application/x-www-form-urlencoded
app.use(express.urlencoded({ extended: true }));
// parse application/json
app.use(express.json());

const multer = require('multer')
app.use(multer().any())

app.get('/login', (req, res)=>{
    res.sendFile(path.join(__dirname, 'html', 'login.html'))
})
app.get('/register', (req, res)=>{
    res.sendFile(path.join(__dirname, 'html', 'register.html'))
})

app.post('/login', (req, res)=>{
    res.json(req.body)
})
app.post('/register', (req, res)=>{
    res.json(req.body)
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