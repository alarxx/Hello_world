/**
  Этот компонент, загружает данные пользователя Github
  и сохраняет их в локальной памяти браузера в усеченном виде.
  В следующий раз, когда потребуется этот пользователь, он просто
  загрузит данные из локальной памяти браузера.

  fetch идет синхронно, но происходит после рендера основной страницы.
*/

import React, {useState, useEffect} from 'react';


const loadJSON = key =>
  key && JSON.parse(localStorage.getItem(key));

const saveJSON = (key, data) =>
  localStorage.setItem(key, JSON.stringify(data));


export default function Save({login = 'Alar-q'}){
  // Ищем data в памяти
  const [data, setData] = useState(loadJSON(`user:${login}`));

  // Если не нашли загружаем
  // После этого хука, у нас ререндерится страница
  useEffect(()=>{
    if(!data){
      fetch(`https://api.github.com/users/${login}`)
        .then(res => res.json())
        .then(setData)
        .catch(console.error);
    }
  }, [login]);// Первый раз и при смене login

  // Если загрузили сохраняем в память
  useEffect(()=>{
    if(data){
      saveJSON(`user:${login}`, {
        name: data.name,
        login: data.login,
        avatar_url: data.avatar_url,
        location: data.location
      });
    }
  }, [data]);// Первый раз и при изменении data

  // useEffect(()=>localStorage.clear());

  // Загрузили, сохранили, выводим результат на страницу
  if(data)
    return <pre>{JSON.stringify(data, null, 2)}</pre>

  return ;
}
