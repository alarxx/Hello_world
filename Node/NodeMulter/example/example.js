const path = require("path");

require('dotenv').config()

const express = require('express');

const app = express();

/** JSON Parsing */
// parse application/x-www-form-urlencoded
app.use(express.urlencoded({ extended: true }));
// parse application/json
app.use(express.json());

/** File Upload multipart/form-data */
const multer = require('multer')
require('../multer-manager').initialize({ clearTempIntervalTime: 2000 });

const { storage } = require('../multer-manager');

// Для примера
const filters = function(req, file, cb){
    if(file.mimetype === 'image/png' || file.mimetype === 'image/jpeg'){
        cb(null, true)
    }
    else {
        cb(null, false)
    }
}

// Для примера
const limits = {
    fileSize: 1024 * 1024 * 40 // 40 megabytes
}

// Для примера
const errorHandler = function (err, req, res, next){
    if (err instanceof multer.MulterError) {
        // A Multer error occurred when uploading.
        res.status(400).json({error: err.message})
    }   else {
        next()
    }
}

const upload = multer({ storage, filters, limits });

const cpUpload = upload.fields([{ name: 'avatar', maxCount: 1 }]);

app.post('/avatar', cpUpload, errorHandler, (req, res)=>{

    console.log(req.files);
    /*
    //Result of req.files:
    [Object: null prototype] {
        avatar: [{
            fieldname: 'avatar',
            originalname: '1.jpg',
            encoding: '7bit',
            mimetype: 'image/jpeg',
            hash: '31710eb89a86bfdbdd472ebba6e9da23c18345a70c4063c43bcdab1ed7412df3',
            path: 'tmp\\files\\f\\f3ccdd27d2000e3f9255a7e3e2c48800',
            size: 135611
        }]
    }
    */

    res.send("file avatar test");
});

app.use((err, req, res, next)=>{
    console.log("Catching error", err)
    res.status(400).json({error: err.message})
})


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



