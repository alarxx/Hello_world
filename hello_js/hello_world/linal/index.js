const Mat = require('./mat.js');
const Core = require('./core.js');

const zeros = new Mat(3, 3);
console.log(zeros.toString("orig"));


const ones = Core.add(zeros, 1);
console.log(ones?.toString("orig + 1"));

const sub = ones.submat(0, 0, 2, 2);
console.log(sub.toString("submat"));

const addmat = Core.add(zeros, Core.add(zeros, 2));
console.log(zeros.toString("addmat"));