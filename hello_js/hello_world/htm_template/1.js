const strong = (fragments, ...values)=>{
	let res = fragments[0]
	for(let i=0; i<values.length; i++)
		res += `<strong>${values[i]}</strong>${fragments[i+1]}`
	return res
}

const person = {name: "Name", age: 15}
let message = strong`Name name ${person.name}${person.age} age`
console.log(message)