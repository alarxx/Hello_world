import React, {useState, useEffect, useRef} from 'react'

export default function Checkbox(){
  const [checked, setChecked] = useState(false);
  const inputTxt = useRef();

  useEffect(()=>{
    alert(`checked is ${checked}`);
    inputTxt.current.focus();
  });

  return (
    <>
      <input
        ref={inputTxt}
        type="text"
      />
      <input
        type="checkbox"
        value={checked}
        onChange={()=>setChecked(!checked)}
      />
      {checked ? "checked" : "not checked"}
    </>
  );
}
