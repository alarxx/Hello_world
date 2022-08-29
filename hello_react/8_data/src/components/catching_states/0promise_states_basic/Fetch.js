import React, {useEffect, useState} from 'react';

const print = (m) => {
	console.log(m);
	return m;
}

export default function Fetch(){
	const [data, setData] = useState();
	const [error, setError] = useState();
	const [loading, setLoading] = useState(false);

	useEffect(()=>{
		setLoading(true);
		fetch('https://api.github.com/users/Alar-q')
			.then(res => res.json())
			.then(print)
			.then(setData)
			.then(()=>setLoading(false))
			.catch(setError);
	}, []);

	if(loading)
		return <h1>loading...</h1>
	else if(error)
		return <pre>{JSON.stringify(error, null, 2)}</pre>
	else if(!data)
		return null;
	else return (
		<div className="githubUser">
			<h1>{data.login}</h1>
			<img
					src={data.avatar_url}
					alt={data.login}
					style={{width: 200}}
			/>
			<div>
				<p>name: {data.name}</p>
				<p>location: {data.location ? data.location : "no information"}</p>
			</div>
		</div>
	);
}
