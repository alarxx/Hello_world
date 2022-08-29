import React, {useEffect, useState} from 'react';

// не проверял. прикол в хеадерах

export default function Post(){
	useEffect(()=>{
		fetch('https://api.github.com/users/Alar-q', {
      method: "GET",
      headers: {
        Authorization: 'Bearer ${token}'
      }
    });
	}, []);

	return (<h1>Hello</h1>);
}
