import React from 'react';
import {FaStar} from "react-icons/fa";

const createArray = (l) => [...new Array(l)];

function Star({selected, onClick = f=>f}){
  return (<FaStar color={selected?"red":"grey"} onClick={onClick}/>);
}

export default function StarRating({totalStars=5, rating=0, onRate = f=>f}){
  return (
    <div>
      {createArray(totalStars).map((n, i) =>
        <Star
          key={i}
          selected={rating > i}
          onClick={()=>onRate(i+1)}
        />
      )}
      <p>{rating} of {totalStars} stars</p>
    </div>
  );
}
