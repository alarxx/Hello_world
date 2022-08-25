import React from 'react';
import Rating from './Rating.js';
import {FaTrash} from 'react-icons/fa'

export default function Color({id, title, color, rating, onRemove = (f)=>f, onRate = (f)=>f}){
  return (
    <section>
      <h4>{title}</h4>
      <button onClick={()=>onRemove(id)}>
        <FaTrash/>
      </button>
      <div style={{height: 50, width: 50, backgroundColor: color}} />
      <Rating rating={rating} onRate={rating => onRate(id, rating)}/>
    </section>
  );
}
