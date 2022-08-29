import React, {useState, useCallback} from 'react';
import UserRepos from './components/UserRepos.js';
import SearchForm from './components/SearchForm.js';
import GithubUser from './components/GithubUser.js';


export default function Main(){
  const [login, setLogin] = useState("");

  return (
    <>
      <SearchForm onSearch={setLogin}/>
      {login &&
        <>
          <GithubUser login={login}/>
          <UserRepos
            login={login}
            onSelect={repo=>console.log(`${repo} selected`)}
          />
        </>
      }
    </>
  );
}


/**
  Ошибка происходит при первом рендеринге, так как изначально у нас login="".
  Не могу понять почему не выходит сообщение об ошибке 404 на экране,
  а выходит компонент renderSuccess, как будто ошибки и не было.
  Можно жестко запрограммировать при определенных сообщениях выкидывать ошибку!?
*/
