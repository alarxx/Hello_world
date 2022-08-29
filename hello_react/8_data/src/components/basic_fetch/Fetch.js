import React, {useEffect, useState} from 'react';

export default function Fetch(){
	const [val, setVal] = useState();

	useEffect(()=>{
		fetch('https://api.github.com/users/Alar-q')
			.then(res => res.json())
			.then(res => {
					setVal(res);
					return res;
				})
			.catch(console.error);
	}, []);

	// Не знал про pre, выдумал очень странный способ вывести json
	// {val!=null? Object.keys(val).map((key)=><p>{key}: {val[key]}</p>) :"no"}

	return (
		<>
			<h1>Hello Fetch</h1>
			<pre>{JSON.stringify(val, null, 2)}</pre>
		</>);
}
