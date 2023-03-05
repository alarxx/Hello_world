/** File Upload multipart/form-data */

function log(...str) {
    console.log(...str)
}


const multer = require('multer');
const fs = require("fs");
const path = require("path");
const customStorage = require('./storage/custom-storage');

async function clearTemp({ rootDir, force, clearTempTime }){
    try{
        const files = await fs.promises.readdir(rootDir);

        const date = Date.now();

        await Promise.all(files.map(async file => {
            const filepath = path.join(rootDir, file);
            const stats = await fs.promises.stat(filepath);

            log(file, stats.isFile() ? 'file' : 'directory');

            if(stats.isDirectory()){
                // Проверяем, что папка создана больше n времени назад.
                // Название каждой папки - время создания в миллисекундах, но лучше использовать время создания напрямую.
                // const expired = date - stats.birthtimeMs >= clearTempTime;
                // log(expired);
                if(force || date - stats.birthtimeMs >= clearTempTime){
                    // fs.rm recursive force удаляет все в папке и не выводит ошибок.
                    // Вместо fs.rm аналогично можно было бы рекурсивно пользоваться fs.rmdir и fs.unlink.
                    await fs.promises.rm(filepath, { recursive: true, force: true });
                }
            }
        }))
    }
    catch(err){
        log(err);
    }
}


const object = {}

/**
 * @clearTempTime - время в миллисекундах, по истечению которого временный файл удаляются (по ум. 5 часов).
 * @clearTempIntervalTime - время в миллисекундах, частота проверки истечения срока жизни временных файлов (по ум. 1 сек).
 * @rootDir - месторасположение временных файлов (по ум. "tmp/files/").
 * */
function initialize(opts={}){

    // Default values
    if(!opts.clearTempTime){
        opts.clearTempTime = 1000 * 60 * 60 * 5; // 5 hours
    }
    if(!opts.clearTempIntervalTime){
        opts.clearTempIntervalTime = 1000; // 1 second
    }
    if(!opts.rootDir){
        opts.rootDir = path.join(__dirname, 'tmp', 'files');
    }

    fs.mkdir(opts.rootDir, { recursive: true }, (err)=>{});

    /*object.storage = multer.diskStorage({
        destination: function (req, file, cb) {
            const dir = path.join(opts.rootDir, String(Date.now()));
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
    })*/
    object.storage = customStorage();

    // Для примера
    object.imageFileFilter = function(req, file, cb){
        if(file.mimetype === 'image/png' || file.mimetype === 'image/jpeg'){
            cb(null, true)
        }
        else {
            cb(null, false)
        }
    }

    // Для примера
    object.limits_40mb = {
        fileSize: 1024 * 1024 * 40 // 40 megabytes
    }

    // Для примера
    object.errorHandler = function (err, req, res, next){
        if (err instanceof multer.MulterError) {
            // A Multer error occurred when uploading.
            res.status(400).json({error: err.message})
        }   else {
            next()
        }
    }

    // Очистить папку буфера файлов полностью на старте
    clearTemp({ rootDir: opts.rootDir, force: true, clearTempTime: opts.clearTempTime })
    // Каждую секунду проверять истек ли срок хранения какого файла
    const clearTempInterval = setInterval(()=>{
        clearTemp({ rootDir: opts.rootDir, force: false, clearTempTime: opts.clearTempTime })
    }, opts.clearTempIntervalTime)
}


object.initialize = initialize;

// Примеры multer middleware
// const upload = multer({ storage, fileFilter, limits })
// const upload_middleware = upload.fields([{ name: 'avatar,' maxCount: 1 }]);
// const upload_middleware = upload.any(); // В принципе можно использовать any всегда, так делает express-fileupload


module.exports = object;
