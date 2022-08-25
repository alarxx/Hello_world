import React, {memo, useState, useCallback} from 'react';

// Чистый компонент
function Cat({name, meow = f => f}){
  console.log(`rendering ${name}`);
  return <p onClick={()=>meow(name)}>{name}</p>;
}
// Можно мемоизировать meow, как в MainWCallback,
// тогда ссылка функции всегда будет одна и та же, name - примитив
const PureCat = memo(Cat, (prev, next)=>prev.name===next.name);

export default function Main(){
  const [cats, setCats] = useState(["Biscuit", "Jungle", "Outlaw"]);


  return (
    <>
      <div>
        {cats.map((name, i)=><PureCat key={i} name={name} meow={(name)=>console.log(`${name} has meowed`)}/>)}
      </div>

      <button onClick={()=>setCats([...cats, prompt("Name a cat")])}>
        Add a Cat
      </button>
    </>
  );
}
