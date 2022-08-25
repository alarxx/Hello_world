import React from 'react';
import {FaTrash} from 'react-icons/fa'

import {useColors} from './color-hooks.js'
import StarRating from './StarRating.js'

export default function Color({id, title, color, rating}){
  const {rateColor, removeColor} = useColors();
  return (
    <section>
      <h4>{title}</h4>

      <button onClick={()=>removeColor(id)}>
        <FaTrash/>
      </button>

      <div style={{height: 50, width: 50, backgroundColor: color}} />

      <StarRating rating={rating} onRate={rating => rateColor(id, rating)}/>
    </section>
  );
}
