import React, {useState, useEffect} from 'react';
import useIterator from '../hooks/useIterator.js';

export default function RepoMenu({
    repositories=[{name: "Alar-q"}, {name:"Alar"}, {name:"mojang"}],
    onSelect = f => f
  }){
  const [item, prev, next] = useIterator(repositories);
  const name = item?.name;

  useEffect(()=>{
    if(!name) return;
    onSelect(name);
  }, [name]);

  return (
    <div style={{display: "flex"}}>
      <button onClick={prev}>&lt;</button>
      <p>{name}</p>
      <button onClick={next}>&gt;</button>
    </div>
  );
}
