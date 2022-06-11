function Employee(name, salary){
	this.name = name;
	this.salary = salary;
}
Employee.prototype.raiseSalary = function(percent){
	this.salary *= 1 + percent/100;
}

const harry = new Employee("Harry", 190);
harry.raiseSalary(100);

console.log(harry);
