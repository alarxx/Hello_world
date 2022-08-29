import React, {useState, useEffect} from 'react';

export default function useFetch(uri){
  const [data, setData] = useState();
  const [error, setError] = useState();
  const [loading, setLoading] = useState(true);

  useEffect(()=>{
    if(!uri) return;
    setLoading(true);

    (async () => {
      try{
        const response = await fetch(uri);
        const userData = await response.json();
        console.log(userData);
        setLoading(false);
        setData(userData);

        // Костыль! Почему-то не выходит ошибка
          // if(data?.message==="Not Found")
          //   throw data;

      }
      catch(err){
        setError(err);
        console.error(err);
      }
    })();

  }, [uri]);

  return {loading, data, error};
}
