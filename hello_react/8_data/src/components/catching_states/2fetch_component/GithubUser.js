/**
  Основная идея понятна - вынести
  логику обработки состояний fetch
  в отдельный компонент Fetch.
*/

import React from 'react';
import Fetch from './Fetch.js'

function userDetails(data){
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

function fallback(error){
  return (
    <p>
      Something went wrong... {error.message}
    </p>
  );
}

function loadingSpinner(){
  return (<h1>Loading spinner...</h1>);
}

export default function GithubUser({login}){
  return (
    <Fetch
      uri={`https://api.github.com/users/${login}`}
      renderSuccess={userDetails}
      loadingFallback={loadingSpinner}
      renderError={fallback}
    />);

}
