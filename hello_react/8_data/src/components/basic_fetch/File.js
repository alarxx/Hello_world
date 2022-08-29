import React, {useEffect, useState} from 'react';

//Для загрузки файлов требуется HTTP-запрос multipart-formdata

const imgFile = {};

export default function Post(){
	useEffect(()=>{
    const formData = new FormData();
    formData.append("username", "Alar-q");
    formData.append("fullname", "Akilbekov Alar");
    formData.append("avatar", imgFile);

		fetch('/create/user', {
      method: "POST",
      body: formData
    })
	}, []);

	return (<h1>Hello</h1>);
}
