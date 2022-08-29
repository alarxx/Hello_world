import React from 'react';


const getStatusFast = () => "success - ready";

const getErrorStatus = () => {
  throw new Error("something went wrong")
};

export default function Status(){
  const status = fetch('https://pokeapi.co/api/v2/ability/').then(res=>res.json());
  throw status
  return (
      <pre>{JSON.stringify(status, null, 2)}</pre>
  );
}
