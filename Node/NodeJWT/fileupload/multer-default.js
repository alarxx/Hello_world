const multer = require('multer');

const multerManager = require('./multer-manager');

multerManager.initialize();

const upload = multer({ storage: multerManager.storage });

module.exports = function(){
    return function(req, res, next){
        upload.any()(req, res, function(){
            const files = {};
            req.files.map(file => {
                files[file.fieldname] = file;
            })
            req.files = files;
            next();
        })
    }
}
