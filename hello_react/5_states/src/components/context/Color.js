import React from 'react';
import {FaTrash} from 'react-icons/fa'

import StarRating from './StarRating.js'

export default function Color({id, title, color, rating, onRemove = f=>f, onRate = f=>f}){
  return (
    <section>
      <h4>{title}</h4>

      <button onClick={()=>onRemove(id)}>
        <FaTrash/>
      </button>

      <div style={{height: 50, width: 50, backgroundColor: color}} />

      <StarRating rating={rating} onRate={rating => onRate(id, rating)}/>
    </section>
  );
}
