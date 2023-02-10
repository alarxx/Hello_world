// Обновление токена
// Должен генерировать новый AC and RT

const express = require('express');
const passport = require('passport');

const Router = express.Router();

const tokens = require(require.main.path + '/auth/tokens');

// Ему же не нужен аксес токен, просто сюда переходим и возвращается новый аксес токен, используются cookie
Router.get('/', tokens.refresh());

Router.delete('/', tokens.delete());

// TEST
Router.get('/protected', passport.authenticate('jwt', { session: false }), (req, res)=>{
  res.json({
    user: req.user
  });
});

module.exports = Router;
