import React from 'react';
import Rating from './Rating.js';

export default function Color({title, color, rating}){
  return (
    <section>
      <h4>{title}</h4>
      <div style={{height: 50, width: 50, backgroundColor: color}} />
      <Rating rating={rating}/>
    </section>
  );
}
