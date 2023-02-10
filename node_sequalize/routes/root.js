const express = require('express');
const Router = express.Router();

Router.get('/', require('../controllers/root'));

Router.use((err, req, res, next)=>{
    res.json({message: err.message});
});

module.exports = Router;
