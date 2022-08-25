import React, {useState, useReducer} from 'react';

export default function Main(){
  const [checked, setChecked] = useState(false);

  const toggle = () => {
    setChecked(checked => !checked);
  }

  return (<>
      <input
        type="checkbox"
        value={checked}
        onChange={toggle}
      />
      <h2>{checked ? "checked" : "not checked"}</h2>
    </>);
}
