const express = require('express');
const Router = express.Router();

Router.get('/', (req, res)=>{
    res.json({
        page: 'home',
        session: req.session,
        email: req.user.email
    });
});

Router.use('/auth', require('./auth/auth'))
Router.use('/api', require('./api/api'))

Router.use((error, req, res, next)=>{
    console.log('root route catched error');
    res.json({error});
});

module.exports = Router;