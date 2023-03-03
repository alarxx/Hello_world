/** File Upload multipart/form-data */

function log(...str) {
    console.log(...str)
}


const multer = require('multer');
const fs = require("fs");
const path = require("path");


async function clearTmp({ tmpDir, force, clearTmpTime }){
    try{
        const files = await fs.promises.readdir(tmpDir);

        const date = Date.now();

        await Promise.all(files.map(async file => {
            const filepath = path.join(tmpDir, file);
            const stats = await fs.promises.stat(filepath);

            log(file, stats.isFile() ? 'file' : 'directory');

            if(stats.isDirectory()){
                // Проверяем, что папка создана больше n времени назад.
                // Название каждой папки - время создания в миллисекундах, но лучше использовать время создания напрямую.
                // const expired = date - stats.birthtimeMs >= clearTmpTime;
                // log(expired);
                if(force || date - stats.birthtimeMs >= clearTmpTime){
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
 * @clearTmpTime - время в миллисекундах, по истечению которого временный файл удаляются (по ум. 5 часов).
 * @clearTmpIntervalTime - время в миллисекундах, частота проверки истечения срока жизни временных файлов (по ум. 1 сек).
 * @tmpDir - месторасположение временных файлов (по ум. "tmp/files/").
 * */
function initialize({
                        clearTmpTime = 1000 * 60 * 60 * 5, // 5 hours
                        clearTmpIntervalTime = 1000, // 1 second
                        tmpDir = path.join(__dirname, 'tmp', 'files')
}){

    fs.mkdir(tmpDir, { recursive: true }, (err)=>{})

    object.storage = multer.diskStorage({
        destination: function (req, file, cb) {
            const dir = path.join(tmpDir, String(Date.now()));
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
    clearTmp({ tmpDir, force: true, clearTmpTime })
    // Каждую секунду проверять истек ли срок хранения какого файла
    const clearTmpInterval = setInterval(()=>{
        clearTmp({ tmpDir, force: false, clearTmpTime })
    }, clearTmpIntervalTime)
}

// Перемещение файла в другую директорию. Закрепление файла из буфера
object.move = async function(file, { dir=path.join(__dirname, 'data', 'files') }){

}

// Удаление файла в буфере. Можно не пользоваться, все равно через определенное время файл удалится автоматически.
object.clear = async function(file){

}


object.initialize = initialize;

// Примеры multer middleware
// const upload = multer({ storage, fileFilter, limits })
// const upload_middleware = upload.fields([{ name: 'avatar,' maxCount: 1 }]);
// const upload_middleware = upload.any(); // В принципе можно использовать any всегда, так делает express-fileupload


module.exports = object;
