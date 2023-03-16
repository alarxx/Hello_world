const Router = require('express').Router();

const userController = require('../../controllers/user-controller')

const { body } = require('express-validator');

const validation = require('../../middlewares/validation')

Router.post('/registration',
    body('email').isEmail(),
    body('password').isLength({ min: 3, max: 32 }),
    validation(),
    userController.registration)
Router.post('/login', userController.login)
Router.post('/logout', userController.logout)
Router.get('/activate/:link', userController.activate)
Router.get('/refresh', userController.refresh)
Router.get('/users', userController.getUsers)

module.exports = Router;