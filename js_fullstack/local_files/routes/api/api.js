const express = require('express');
const Router = express.Router();

Router.use(require.main.require('./auth/checkAuthenticated'));

Router.get('/', (req, res)=>{
    res.json({page: 'api'});
});

Router.use('/post', require('./post/post'))

module.exports = Router;