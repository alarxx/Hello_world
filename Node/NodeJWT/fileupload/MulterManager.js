/** File Upload multipart/form-data */

function log(...str) {
    console.log("MulterManager.js", ...str)
}

const fs = require("fs");
const path = require("path");
const customStorage = require('./custom-storage');

/**
 * @clearTmpTime - время в миллисекундах, по истечению которого временный файл удаляются (по ум. 5 часов).
 * @clearTmpIntervalTime - время в миллисекундах, частота проверки истечения срока жизни временных файлов (по ум. 1 сек).
 * @tmpDir - месторасположение временных файлов (по ум. "tmp/").
 * @dstDir - месторасположение постоянных файлов (по ум. "data/").
 * */
module.exports = class MulterManager {
    constructor(opts={}) {
        log("Initialize Multer Manager");

        this.clearTmpTime = opts.clearTmpTime || 1000 * 60 * 60 * 5; // 5 hours
        this.clearTmpIntervalTime = opts.clearTmpIntervalTime || 1000 * 60 * 10; // 10 minute
        this.tmpDir = opts.tmpDir || path.join('tmp');
        this.dstDir = opts.dstDir || path.join('data');

        fs.promises.mkdir(this.tmpDir, { recursive: true }).catch(err=>{});
        fs.promises.mkdir(this.dstDir, { recursive: true }).catch(err=>{});

        this.storage = customStorage(this);
    }

    startClearInterval(){
        // Очистить папку буфера файлов полностью на старте
        this.clearTemp({ force: true });
        const that = this;
        // Каждую N времени проверять истек ли срок хранения какого файла
        this.clearTmpInterval = setInterval(() => {
            that.clearTemp({ force: false })
        }, this.clearTmpIntervalTime);
    }
    stopClearInterval(){
        // Очистить папку буфера файлов полностью
        this.clearTemp({ force: true });
        clearInterval(this.clearTmpInterval)
    }

    async clearTemp({ force }){
        try{
            const files = await fs.promises.readdir(this.tmpDir);

            const date = Date.now();

            await Promise.all(files.map(async file => {
                const filepath = path.join(this.tmpDir, file);
                const stats = await fs.promises.stat(filepath);


                if(stats.isDirectory()){
                    const expired = date - stats.birthtimeMs >= this.clearTmpTime;

                    // log(file, stats.isFile() ? 'file' : 'directory', force || expired ? 'delete' : '');

                    // Проверяем, что папка создана больше n времени назад.
                    // Название каждой папки - время создания в миллисекундах, но лучше использовать время создания напрямую.

                    if(force || expired){
                        // fs.rm recursive force удаляет все в папке и не выводит ошибок.
                        // Вместо fs.rm аналогично можно было бы рекурсивно пользоваться fs.rmdir и fs.unlink.
                        await fs.promises.rm(filepath, { recursive: true, force: true });
                        log("delete", filepath)
                    }
                }
            }))
        }
        catch(err){
            log(err);
        }
    }
}


