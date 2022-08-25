import React, {useState, useEffect, useMemo} from 'react';

const useAnyKeyToReRender = () => {
  const [, forceRender] = useState();

  useEffect(()=>{
    window.addEventListener("keydown", forceRender);
    return () => window.removeEventListener("keydown", forceRender);
  }, []);
}

export default function Main({children = "alar ulan amir"}){
  useAnyKeyToReRender();

  // const words = ["gnra"]; // после рендера новая ссылка

  // Ссылка не обновляется после каждого рендеринга
  const words = useMemo(()=>children.split(" "), [children]);

  useEffect(()=>{
    console.log(`${words}`);
  }, [words]); // проверяет ссылку, если не примитив

  useEffect(()=>console.log("render"));

  return (<h1>{children}</h1>);
}
