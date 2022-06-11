const events = require('events');

class Logger extends events{
    log(arg) {
        console.log(arg);
        this.emit('mes', 'arg');
    }
}

module.exports = Logger;