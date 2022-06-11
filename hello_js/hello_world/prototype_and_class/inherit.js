class Animal {
	#name;
	constructor(name){
		this.#name = name;
	}

	toString(){
		return this.#name;
	}
}

class Lion extends Animal {
	#lines;
	#name;
	constructor(name, lines){
		super(name);
		this.#lines = lines;
		this.#name = "Lions variable";
	}

	getname(){
		return this.#name;
	}
}

const lion = new Lion("LionName AnimalClass", 123);
console.log(lion.toString());
console.log(lion.getname());