const MulterManager = require("./multer-manager");
const multer = require("multer");

module.exports = () => {
    const multerManager = new MulterManager();

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