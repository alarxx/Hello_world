import React, {useState} from 'react';

import GithubUser from './GithubUser.js'
import SearchForm from '../1fetch_hook/SearchForm.js'

export default function App(){
  const [login, setLogin] = useState("");
  return (
    <div style={{textAlign: 'center'}}>
      <SearchForm onSearch={setLogin}/>
      <div style={{marginTop: '3em'}}>
        <GithubUser login={login}/>
      </div>
    </div>
  );
}
