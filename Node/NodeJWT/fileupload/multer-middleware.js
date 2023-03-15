const MulterManager = require("./MulterManager");
const multer = require("multer");

module.exports = (opts) => {
    const multerManager = new MulterManager(opts);
    multerManager.startClearInterval();

    const upload = multer({ storage: multerManager.storage });

    return (req, res, next) => {
        upload.any()(req, res, () => {
            const files = {};
            req.files.map(file => {
                files[file.fieldname] = file;
            })
            req.files = files;
            next();
        })
    }
}