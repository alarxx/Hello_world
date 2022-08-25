
//Render React
//useLayoutEffect
//Render DOM
//useEffect


import React, {useState, useLayoutEffect, useEffect} from 'react';

/*Кажется пример с размерами плохо отображает суть useLayoutEffect
  Как будто без listener-ов было бы лучше */

function useWindowSize(){
  const [width, setWidth] = useState(0);
  const [height, setHeight] = useState(0);

  const resize = () => {
    setWidth(window.innerWidth);
    setHeight(window.innerHeight);
    // width height назначаются после завершения хука useWindowSize
    console.log("window", window.innerWidth, window.innerHeight);
  };

  useLayoutEffect(()=>{
    window.addEventListener("resize", resize);
    resize(); // Вот как тут, без листнеров
    return () => window.removeEventListener("resize", resize);
  }, []);

  return [width, height]
}


function useMousePosition(){
  const [x, setX] = useState(0);
  const [y, setY] = useState(0);

  const setPosition = ({x, y})=>{
    //как обратиться к наружнему х и у
    setX(x);
    setY(y);
    console.log("mouse", x, y);
  };

  useLayoutEffect(()=>{
    window.addEventListener("mousemove", setPosition);
    return () => window.removeEventListener("mousemove", setPosition);
  }, []);

  return [x, y];
}


export default function Main(){
  const [w, h] = useWindowSize();
  const [x, y] = useMousePosition();

  return (<>
        <h1>window {w} {h}</h1>
        <h1>mouse {x} {y}</h1>
      </>);
}
