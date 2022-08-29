/**
  Основная идея понятна, вынести
  логику обработки состояний fetch
  в отдельную функцию-хук.
*/

import React from 'react';
import useFetch from './useFetch.js'

export default function GithubUser({login}){
  const {loading, data, error} =
    useFetch(`https://api.github.com/users/${login}`);

  if(loading)
    return <h1>loading...</h1>
  if(error)
    return <pre>{JSON.stringify(error, null, 2)}</pre>

  return (
    <div className="githubUser">
      <img src={data.avatar_url} alt={data.login} width={200}/>
      <div>
        <h1>{data.login}</h1>
        <p>name: {data.name}</p>
        <p>location: {data.location}</p>
      </div>
    </div>
  );
}
