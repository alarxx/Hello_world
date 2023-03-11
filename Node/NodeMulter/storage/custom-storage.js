const fs = require('fs')
const path = require('path')
const crypto = require('crypto')

function CustomStorage (opts){
  this.tmpDir = opts.tmpDir;
}

/**
 * Записывает входящий файл во временную папку.
 * */
CustomStorage.prototype._handleFile = function _handleFile (req, file, cb) {
  (async ()=>{
    const user = { id: '6405e8c268dadbbfadd21932' }; // from req, можно, наверное, еще в body какие нибудь данные передавать
    const hashBuilder = crypto.createHash('md5');

    // В буфер кладем хэш именно по названию, 
    // ну и данные пользователя, потому что названия у разных клиентов могут совпасть, 
    // а при переносе уже пользуемся хэшем полного файла

    // !!! Нужна проверка на существование такого же файла !!! Возможно клиент будет скидывать 2 одинаковых файла одновременно
    // const nameHash = crypto.createHash('md5').update(`${user.id}${file.originalname}`).digest('hex');

    const filename = `${user.id}-${file.originalname}`;

    const destination = path.join(this.tmpDir, String(Date.now()));
    await fs.promises.mkdir(destination, { recursive: true }).catch(err => {});

    const tmpPath = path.join(destination, filename);
    const outStream = fs.createWriteStream(tmpPath)

    file.stream.pipe(outStream)

    file.stream.on('data', async chunk => {
      // outStream.write(chunk)
      hashBuilder.update(chunk)
    })

    outStream.on('error', cb)

    outStream.on('finish', function () {
      cb(null, {
        hash: `${hashBuilder.digest('hex')}-${user.id}`,
        tmpPath,
        size: outStream.bytesWritten
      })
    })
  })()
}

CustomStorage.prototype._removeFile = function _removeFile (req, file, cb) {}

module.exports = function (opts) {
  return new CustomStorage(opts)
}
