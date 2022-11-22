const path = require('path');
const fs = require('fs');

const dataDir = path.resolve(__dirname, '..', 'data');
const photosDir = path.join(dataDir, 'photos');

if(!fs.existsSync(dataDir)) fs.mkdirSync(dataDir);
if(!fs.existsSync(photosDir)) fs.mkdirSync(photosDir);

const lol = 0;
console.log('how many times this code reads by interpreter', lol);
lol++;
// const {promisify} = require('util');
// const mkdir = promisify(fs.mkdir);
// const rename = promisify(fs.rename);

exports = async (photo)=>{
	const dir = photosDir + '/' + Date.now();
	const path = dir + '/' + photo.originalFilename;
	await async () => fs.mkdir(dir);
	await async () => fs.rename(photo.path, path);
};