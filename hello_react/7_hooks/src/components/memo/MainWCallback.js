import React, {memo, useState, useCallback} from 'react';

// Чистый компонент
function Cat({name, meow = f => f}){
  console.log(`rendering ${name}`);
  return <p onClick={()=>meow(name)}>{name}</p>;
}

const PureCat = memo(Cat);

export default function Main(){
  const [cats, setCats] = useState(["Biscuit", "Jungle", "Outlaw"]);

  const meow = useCallback(name => console.log(`${name} has meowed`), []);

  return (
    <>
      <div>
        {cats.map((name, i)=><PureCat key={i} name={name} meow={meow}/>)}
      </div>

      <button onClick={()=>setCats([...cats, prompt("Name a cat")])}>
        Add a Cat
      </button>
    </>
  );
}
