const path = require("path");

require('dotenv').config()

const express = require('express');

const app = express();

/** CORS */
// app.use(require('cors')())

/** JSON Parsing */
// parse application/x-www-form-urlencoded
app.use(express.urlencoded({ extended: true }));
// parse application/json
app.use(express.json());

/** Static files */
app.use(express.static(path.join(__dirname, 'public')))

/** File Upload multipart/form-data */
const multer = require('multer')
require('./multer-manager').initialize({})
const {storage, limits_40mb: limits, imageFileFilter: fileFilter, errorHandler} = require('./multer-manager');

const upload = multer({ storage, limits, fileFilter })
const cpUpload = upload.fields([{ name: 'avatar', maxCount: 1 }]);
app.post('/avatar', cpUpload, errorHandler, (req, res)=>{
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


const PORT = process.env.PORT || 3000;
app.listen(PORT, ()=>{
    console.log(`server is listening on port ${PORT}`)
});

/*const fs = require('fs')
const path = require("path");

const currentPath = path.join(__dirname, "dataTmp", "meme.jpg");
const destinationPath = path.join(__dirname, "data", "meme.jpg");

// Нужно написать move функцию под multer
function move(currentPath, destinationPath){
    fs.rename(currentPath, destinationPath, function (err) {
        if (err) {
            console.log(err)
        } else {
            console.log("Successfully moved the file!");
        }
    });
}

move(currentPath, destinationPath)*/



