/** File Upload multipart/form-data */

function log(...str) {
    console.log(...str)
}

const fs = require("fs");
const path = require("path");
const customStorage = require('./storage/custom-storage');


const object = {}

async function _clearTemp({ force, tmpDir, clearTempTime }){
    try{
        const files = await fs.promises.readdir(tmpDir);

        const date = Date.now();

        await Promise.all(files.map(async file => {
            const filepath = path.join(tmpDir, file);
            const stats = await fs.promises.stat(filepath);


            if(stats.isDirectory()){
                const expired = date - stats.birthtimeMs >= clearTempTime;

                // log(file, stats.isFile() ? 'file' : 'directory', force || expired ? 'delete' : '');

                // Проверяем, что папка создана больше n времени назад.
                // Название каждой папки - время создания в миллисекундах, но лучше использовать время создания напрямую.

                if(force || expired){
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

async function _fileExists(filepath){
    try{
        await fs.promises.access(filepath, fs.constants.F_OK)
        return true;
    }catch(e){
        return false;
    }
}

function _saveFileWrapper({ dstDir }){
    /**
     * returns a new file path, but returns null if the same file already exists.
     * Может выйти ошибка из fs.rename, если file.path в tmp почему-то не существует.
     * */
    return async function saveFile(file){
        // Нужно переместить из file.tmpPath в dstDir/hash[0]/hash
        const dir = path.join(dstDir, file.hash.substring(0, 1));
        await fs.promises.mkdir(dir).catch(err=>{});

        // файлы с одинаковым содержимым, будут иметь одинаковый hash
        file.dstPath = path.join(dir, file.hash)

        if(!await _fileExists(file.dstPath)){
            // Может выйти ошибка, если file.path в tmp почему-то не существует.
            // Когда такое может быть? Если юзер залил 2 файла с одинаковыми названиями в одно и то же время в мс
            await fs.promises.rename(file.tmpPath, file.dstPath);
            return true;
        }

        return false;
    }
}
function _deleteFileWrapper(){
    return async function deleteFile(file){
        // Нужно удалить из path
        await fs.promises.unlink(file.tmpPath)
        delete file.tmpPath
    }
}

/**
 * @clearTempTime - время в миллисекундах, по истечению которого временный файл удаляются (по ум. 5 часов).
 * @clearTempIntervalTime - время в миллисекундах, частота проверки истечения срока жизни временных файлов (по ум. 1 сек).
 * @tmpDir - месторасположение временных файлов (по ум. "tmp/files/").
 * */
object.initialize = function(opts={}){
    // Default values
    if(!opts.clearTempTime){
        opts.clearTempTime = 1000 * 60 * 60 * 5; // 5 hours
    }
    if(!opts.clearTempIntervalTime){
        opts.clearTempIntervalTime = 1000 * 60; // 1 minute
    }
    if(!opts.tmpDir){
        opts.tmpDir = path.join('tmp');
    }
    if(!opts.dstDir){
        opts.dstDir = path.join('data');
    }

    fs.promises.mkdir(opts.tmpDir, { recursive: true }).catch(err=>{});
    fs.promises.mkdir(opts.dstDir, { recursive: true }).catch(err=>{});

    // Очистить папку буфера файлов полностью на старте
    _clearTemp({ force: true, ...opts });

    // Каждую секунду проверять истек ли срок хранения какого файла
    const clearTempInterval = setInterval(()=>{
        _clearTemp({ force: false, ...opts })
    }, opts.clearTempIntervalTime);


    object.saveFile = _saveFileWrapper({ ...opts });

    object.deleteFile = _deleteFileWrapper();

    object.storage = customStorage(opts);
}

// Примеры multer middleware
// const upload = multer({ storage, fileFilter, limits })
// const upload_middleware = upload.fields([{ name: 'avatar,' maxCount: 1 }]);
// const upload_middleware = upload.any(); // В принципе можно использовать any всегда, так делает express-fileupload

object.any = function (files){
    files.forEach(file => {
        files[file.fieldname] = file;
    })
}

module.exports = object;
