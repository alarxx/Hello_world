import React, {useEffect, useState} from 'react';

export default function Post(){

	useEffect(()=>{
		fetch('/create/user', {
      method: "POST",
      body: JSON.stringify({username, password, bio})
    })
	}, []);

	return (<h1>Hello</h1>);
}
