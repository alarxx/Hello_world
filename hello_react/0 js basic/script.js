const htmlModule = true;


//DOM
if(htmlModule){
	console.log("DOM");

	const container = document.createElement("container");
	document.body.appendChild(container);

	for (let i = 0; i < 5; i++) {
		let div = document.createElement("div");
		div.onclick = function() {
			alert("This is box #: " + i);
		};
		container.appendChild(div);
	}
}


//Деструктуризация
console.log("Destructuring");

//Объектов
const ob = {
	name: "Alar",
	age: 18,
};
let {name} = ob; //let name = ob.name;
name = "Alina"; // 
console.log(name, ob.name);

//В функциях
const lordify = ({name}) => console.log(`${name} of Canterbury`);
lordify(ob);

//Массивов
const [, , t] = [1, 2, 3];
console.log(t);

//Spread Syntax
const peaks = [1, 2, 3]
const [spread] = [...peaks].reverse();
console.log("spread", spread);


//Inheritance
console.log("Inheritance");

const employeePrototype = {
	raiseSalary: function(percent){
		this.salary *= percent;
	}
}

// Фабричный метод с Прототипным наследованием
function createEmployee(name, salary){
	const emp = {name, salary};
	Object.setPrototypeOf(emp, employeePrototype);
	return emp;
}

// Конструирующая функция
function Employee(name, salary){
	this.name = name;
	this.salary = salary;
}

const emp = createEmployee("Alar", 10);
const emp2 = new Employee("Alina", 20);
console.log(emp);
console.log(emp2);

//class
class Vacation{
 	// v = 10;
	constructor(args){
		this.v = 20;
		this.args = args;
	}
	print(){
		console.log(this.args, this.v);
	}
}

const vac = new Vacation("Vacation");
vac.print();

//Enheritance
class Expedition extends Vacation{
	constructor(args, gear){
		super(args);
		this.gear = gear;
	}

	print(){
		super.print();
		console.log(this.gear);
	}
}

const exp = new Expedition("Expedition", "Gear");
exp.print();

//html js module example 
if(htmlModule){
	console.log(`from other module: ${v}`);
}