import React, {useState, useEffect, useRef} from 'react'

export default function DependenciesArray(){
  const [val, set] = useState("");
  const [phrase, setPhrase] = useState("initial phrase");
  const inputRef = useRef();

  const createPhrase = () => {
    setPhrase(val);
    set("");
  };

  //INITIAL EFFECT []
  useEffect(()=>{
      inputRef.current.focus();
      //DELETE EFFECT
      return ()=>alert("lol");
  }, []);

  useEffect(()=>{
    console.log(`saved phrase: ${phrase}`);
    alert(`saved phrase: ${phrase}`);
  });

  useEffect(()=>{
    console.log(`typing: ${val}`);
  });

  return (
    <>
      <input
        ref={inputRef}
        type="text"
        value={val}
        placeholder={phrase}
        onChange={e => set(e.target.value)}
      />
      <button onClick={createPhrase}>send</button>
    </>
  );
}
