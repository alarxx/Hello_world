import React, {useState, useEffect} from 'react';
import Fetch from './Fetch.js';
import RepoMenu from './RepoMenu.js';

export default function UserRepos({
    login,
    onSelect=f=>f
  }){
  return (
    <Fetch
      uri={`https://api.github.com/users/${login}/repos`}
      renderSuccess={data => (
        <RepoMenu
          repositories={data}
          onSelect={onSelect}
        />
       )}
    />
  );
}
