const path = require("path");

require('dotenv').config()

const express = require('express');

const app = express();

/** JSON Parsing */
// parse application/x-www-form-urlencoded
app.use(express.urlencoded({ extended: true }));
// parse application/json*/
app.use(express.json());

/** Static files */
app.use(express.static(path.join(__dirname, 'public')))

/** File Upload multipart/form-data */
const multer = require('multer');
const fs = require("fs");
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        const dir = `./data/${Date.now()}`;
        fs.mkdir(dir, (err) => {
            if (err) {
                cb(err, null);
            } else {
                cb(null, dir);
            }
        });
    },
    filename: function (req, file, cb) {
        cb(null, file.originalname)
    }
})

function fileFilter(req, file, cb){
    if(file.mimetype === 'image/png' || file.mimetype === 'image/jpeg' || file.mimetype === 'application/zip'){
        cb(null, true)
    }
    else {
        cb(null, false)
    }
}

const limits = {
    fileSize: 1024 * 1024 * 1 // megabytes
}

const upload = multer({ fileFilter, limits, storage })
const cpUpload = upload.fields([{ name: 'avatar', maxCount: 1 }]);
app.post('/avatar', cpUpload, (req, res)=>{
    console.log(req.files);
    res.send("file avatar test")
    /*
    [Object: null prototype] {
        avatar: [
            {
              fieldname: 'avatar',
              originalname: 'routing.png',
              encoding: '7bit',
              mimetype: 'image/png',
              destination: 'data/',
              filename: 'avatar-1677797214541-331342484',
              path: 'data\\avatar-1677797214541-331342484',
              size: 17421
            }
        ]
    }
    */
});

app.get('/', (req, res)=>{
    res.json({page: 'home'})
});

app.post('/', (req, res)=>{
    console.log(req.body);
    res.json(req.body)
});

app.use((err, req, res, next)=>{
    res.status(500).json({error: err.message})
})

const PORT = process.env.PORT || 3000;
app.listen(PORT, ()=>{
    console.log(`server is listening on port ${PORT}`)
});