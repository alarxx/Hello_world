const multerManager = require("./multer-manager");
const multer = require("multer");


module.exports = function(){
    multerManager.initialize();

    const upload = multer({ storage: multerManager.storage });
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
