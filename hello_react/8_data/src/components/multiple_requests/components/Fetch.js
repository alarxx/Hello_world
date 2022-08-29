import React, {useState, useEffect} from 'react';
import useFetch from '../hooks/useFetch.js';

// props компонента Fetch (кроме uri), в моем случае, - это функции,
// которые возращают компоненты или просто делают что-то еще.
// p.s.: в дин. пр. используются только чистые функции,
// loadingFallback не чистая функция, ее лучше сделать компонентом.

export default function Fetch({
    uri,
    renderSuccess = f=>f,
    loadingFallback = () => <p>loading...</p>,
    renderError = (error) => (<pre>{JSON.stringify(error, null, 2)}</pre>)
  }){
    const {loading, data, error} = useFetch(uri);

    // Костыль! Почему-то не выходит ошибка
      // if(data?.message==="Not Found")
      //   setError(data);

    // В книге было показано, что можно передавать в виде
    // свойств компонентов: компоненты без аргументов,
    // компоненты с аргументами, функции; Они называются RenderProps.

    if(loading)
      return loadingFallback(); // loadingFallback - компонент без свойств
    if(error)
      return renderError(error); // renderError(error) - функции
    if(data)
      return renderSuccess(data); // renderSuccess({ data }) - компонент со свойством
}
