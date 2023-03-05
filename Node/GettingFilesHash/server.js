const fs = require("fs");
const path = require("path");

/*
function read(filePath) {
    const readableStream = fs.createReadStream(filePath);

    readableStream.on('error', function (error) {
        console.log(`error: ${error.message}`);
    })

    readableStream.on('data', (chunk) => {
        console.log(chunk);
    })
}

function write(filePath) {
    const writableStream = fs.createWriteStream(filePath);

    writableStream.on('error',  (error) => {
        console.log(`An error occured while writing to the file. Error: ${error.message}`);
    });
}

const from = path.join(__dirname, 'assets', 'meme.jpg')
const to = path.join(__dirname, 'copy', 'meme.jpg')

const readableStream = fs.createReadStream(from);
const writableStream = fs.createWriteStream(to);

readableStream.pipe(writableStream)
readableStream.on('data', (chunk) => {
    console.log("1", chunk);
})
readableStream.on('data', (chunk) => {
    console.log("2", chunk);
})
readableStream.on('data', (chunk) => {
    console.log("3", chunk);
})*/
/*readableStream.on('error', function (error) {
    console.log(`error: ${error.message}`);
})


readableStream.on('end', ()=>{
    writableStream.end()
    console.log('end')
})


writableStream.on('error',  (error) => {
    console.log(`An error occured while writing to the file. Error: ${error.message}`);
});

*/

/*
writableStream.on('finish', () => {
    console.log(`You have successfully created a ${from} copy. The new file name is ${to}.`);
})
*/

const crypto = require('crypto')

const sha1 = path => new Promise((resolve, reject) => {
    const hash = crypto.createHash('sha256')
    const rs = fs.createReadStream(path)
    rs.on('error', reject)
    rs.on('data', chunk => {
        console.log(hash)
        return hash.update(chunk)
    })
    rs.on('end', () => resolve(hash.digest('hex')))
})

sha1(path.join(__dirname, 'assets', '1.exe')).then(hash => {
    console.log(hash)
}).catch(err => {
    console.log(err)
})

/*
setInterval(()=>{
    console.log('fps');
}, 1000)*/
