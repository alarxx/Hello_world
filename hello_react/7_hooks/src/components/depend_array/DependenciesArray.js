import React, {useState, useEffect, useRef} from 'react'

export default function DependenciesArray(){
  const [val, set] = useState("");
  const [phrase, setPhrase] = useState("initial phrase");

  const createPhrase = () => {
    setPhrase(val);
    set("");
  };

  useEffect(()=>{
    alert("Very first render");
    return alert("Very last");
  }, []);

  useEffect(()=>{
    console.log(`typing: ${val}`);
  });

  useEffect(()=>{
    console.log(`saved phrase: ${phrase}`);
  }, [phrase]);



  return (
    <>
      <input
        type="text"
        value={val}
        placeholder={phrase}
        onChange={e => set(e.target.value)}
      />
      <button onClick={createPhrase}>send</button>
    </>
  );
}
