const {validationResult} = require("express-validator");
const ApiError = require("../exceptions/api-error");

// Походу для красоты придется писать свои валидации и для файлов и т.д.
module.exports = () => (req, res, next) => {
    try{
        const errors = validationResult(req);
        if(!errors.isEmpty()){
            return next(ApiError.BadRequest('Ошибка при валидации', errors.array()))
        }
    }
    catch(e){next(e)}
}
