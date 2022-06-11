const events = require('events');

var Logger = require('./log'); 
var logger = new Logger();

logger.on('mes', (arg)=>{
    console.log('called', arg);
});

logger.log('lol');