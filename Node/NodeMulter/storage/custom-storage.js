const fs = require('fs')
// const os = require('os')
const path = require('path')
const crypto = require('crypto')

/*function getFilename (req, file, cb) {
  cb(null, file.originalname); // os.tmpdir());
}
function getDestination (req, file, cb) {
  cb(null, path.join(__dirname, 'tmp', 'files')); // os.tmpdir());
}*/

function CustomStorage (opts) {}

CustomStorage.prototype._handleFile = function _handleFile (req, file, cb) {
  (async ()=>{
    const hash = crypto.createHash('sha256')

    // В буфер кладем хэш именно по названию, а при переносе уже пользуемся
    const nameHash = crypto.createHash('md5').update(file.originalname).digest('hex');

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
        // destination: destination,
        // filename: file.originalname,
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
