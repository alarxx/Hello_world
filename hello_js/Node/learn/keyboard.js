//npm instal readline-sync
const readline = require('readline-sync')

let arr = []

for(let i=0; i<10; i++) arr[i] = i


let arrRes = []

arr.forEach( (a) => arrRes.push(a*a) )

console.log(`hi ${arrRes}`);
