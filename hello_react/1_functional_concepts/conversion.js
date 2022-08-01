// Универсальная функция Reduce

const users = [
		{id: 1, name: "Alar"},
		{id: 2, name: "Alina"},
		{id: 3, name: "Sabina"},
	];

const usernames = users.reduce((res, user)=>{
	return [...res, user.name];
}, []);

const usernames2 = users.map(e => e.name);


const filtered = users.reduce(
	(res, user)=>(
		user.name!="Alar"?[...res, user]:res
	),[]
);
const filtered2 = users.filter(e => e.name != "Alar");

console.log(usernames);
console.log(usernames2);
console.log(filtered);
console.log(filtered2);