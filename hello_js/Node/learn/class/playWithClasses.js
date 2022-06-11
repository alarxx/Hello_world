let Animals = require('./models/Animals.js');
let an = new Animals('Alar', 'human');
console.log(an.kind);

class Human extends Animals{
	constructor(name, kind, skills){
		super(name, kind);
		this.skills = skills;
		this.hui = 0;
	}
	riseHui(n){
		this.hui+=n;
	}
	do(){
		this.skills();
	}
}

let human = new Human('Bumba', 'human', ()=>console.log("Perdun"));

human.do();
human.riseHui(12);
console.log(human.hui);

class Smthng extends Human{
	constructor(obj = {}, ...args){
		super(...args);
		this.per = obj.per || 'no per';
	}

	get name(){
		return this.name;
	}

	set name(name){
		this.name = name;
	}
}

let smt = new Smthng({"per": "lol"}, 'Bumba', 'human', ()=>console.log("Perdun"));
console.log(smt.per);

console.log(smt.get());

