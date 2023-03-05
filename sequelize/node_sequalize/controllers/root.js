const { Sequelize } = require('sequelize');

const { credentials } = require('../config');
console.log(credentials.DB_DNS);

const sequelize = new Sequelize(credentials.DB_DNS);
/*
// Можно еще так подключаться
const sequelize = new Sequelize('database', 'username', 'password', {
    host: 'localhost',
    dialect: /!* 'mysql' | 'mariadb' | 'postgres' | 'mssql' *!/
})
*/

sequelize.authenticate()
    .then(res => {
        console.log('Соединение с БД было успешно установлено')
    })
    .catch(err => {
        console.log('Невозможно выполнить подключение к БД: ', err)
    })

// sequelize.close()

module.exports = async function seq(req, res){
    res.json({message: 'seq'});
}