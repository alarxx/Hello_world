import React, {useState, useReducer} from 'react';

export default function Main(){
  const [number, setNumber] = useReducer((number, newNumber)=>number+newNumber, 0);
  const [checked, toggle] = useReducer(checked => !checked, false);

  return (<>
      <button onClick={()=>setNumber(1)}>add one</button>

      <input
        type="checkbox"
        value={checked}
        onChange={toggle}
      />

      <h1>{number}</h1>

      <h1>{checked ? "checked" : "not checked"}</h1>
    </>);
}
