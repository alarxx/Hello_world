import React, {useEffect, useState} from 'react';

async function requestGithubUser(githubLogin, setUserData){
	try{
		const response = await fetch(
			`https://api.github.com/users/${githubLogin}`
		);
		const userData = await response.json();
		setUserData(userData);
	}catch(err){
		console.error(err);
	}
}

export default function AsyncFetch(){
	const [userData, setUserData] = useState();

	requestGithubUser('Alar-q', setUserData);

	return (
		<>
			{userData!=null ?
				Object.keys(userData).map((key, i) => (<p key={i}>{key}: {userData[key]}</p>) ) 
				: "no"}
		</>);
}
