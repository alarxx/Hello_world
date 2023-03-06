const fs = require('fs')
const path = require('path')
const crypto = require('crypto')

function CustomStorage (opts) {}

CustomStorage.prototype._handleFile = function _handleFile (req, file, cb) {
  (async ()=>{
    const user = { id: '6405e8c268dadbbfadd21932' };
    const hash = crypto.createHash('sha256');

    // В буфер кладем хэш именно по названию, а при переносе уже пользуемся
    const nameHash = crypto.createHash('md5').update(`${user.id}${file.originalname}`).digest('hex');

    const destination = path.join('tmp', 'files', nameHash.substring(0, 1));
    fs.mkdir(destination, { recursive: true }, (err)=>{});

    const finalPath = path.join(destination, nameHash);
    const outStream = fs.createWriteStream(finalPath)

    file.stream.pipe(outStream)

    file.stream.on('data', chunk => hash.update(chunk))

    outStream.on('error', cb)

    outStream.on('finish', function () {
      cb(null, {
        hash: hash.digest('hex'),
        path: finalPath,
        size: outStream.bytesWritten
      })
    })
  })()
}

CustomStorage.prototype._removeFile = function _removeFile (req, file, cb) {
  const path = file.path

  delete file.hash
  delete file.destination
  delete file.filename
  delete file.path

  fs.unlink(path, cb)
}

module.exports = function (opts) {
  return new CustomStorage(opts)
}
