import React from 'react';
import Fetch from './Fetch.js'

function userDetails(data){
  return (
    <div className="githubUser">
      <img src={data.avatar_url} alt={data.login} width={200}/>
      <div>
        <h1>{data.login}</h1>
        {data.name && <p>name: {data.name}</p>}
        {data.location && <p>location: {data.location}</p>}
      </div>
    </div>
  );
}

export default function GithubUser({login}){
  return (
    <Fetch
      uri={`https://api.github.com/users/${login}`}
      renderSuccess={userDetails}
    />);

}
