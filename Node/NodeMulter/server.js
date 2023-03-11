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
const multerManager = require('./multer-manager');
multerManager.initialize();

const { storage, saveFile, deleteFile } = multerManager;

const upload = multer({ storage });
// const cpUpload = upload.fields([{ name: 'avatar', maxCount: 1 }]);

app.use(upload.any(), (req, res, next)=>{
    const files = {};
    req.files.map(file => {
        files[file.fieldname] = file;
    })
    req.files = files;
    next()
})

app.post('/avatar', async (req, res)=>{

    if(!await saveFile(req.files.avatar)){
        await deleteFile(req.files.avatar)
    }

    res.json(req.files.avatar)
    /*{
        "fieldname": "avatar",
        "originalname": "multer-master.zip",
        "encoding": "7bit",
        "mimetype": "application/zip",
        "hash": "cd84a9e79de45a3ed390d4dd1608cd4a-6405e8c268dadbbfadd21932",
        "path": "data\\c\\cd84a9e79de45a3ed390d4dd1608cd4a-6405e8c268dadbbfadd21932",
        "size": 2481661
    }*/
});

app.use((err, req, res, next)=>{
    console.log("Catching error", err)
    res.status(400).json({error: err.message})
})


const PORT = process.env.PORT || 3000;
app.listen(PORT, ()=>{
    console.log(`server is listening on port ${PORT}`)
});



