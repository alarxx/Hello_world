module.exports = (req, res, next) => {
    console.log('custom middleware called');
    next();
};