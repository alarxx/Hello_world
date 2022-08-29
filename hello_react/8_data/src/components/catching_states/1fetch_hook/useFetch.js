import React, {useState, useEffect} from 'react';

const print = (m)=>{
  console.log(m);
  return m;
}

export default function useFetch(uri){
  const [data, setData] = useState();
  const [error, setError] = useState();
  const [loading, setLoading] = useState(true);

  useEffect(()=>{
    if(!uri) return;
    setLoading(true);
    fetch(uri)
      .then(data => data.json())
      .then(print)
      .then(setData)
      .then(()=>setLoading(false))
      .catch(setError);
  }, [uri]);

  return {loading, data, error};
}
