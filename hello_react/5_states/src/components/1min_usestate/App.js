import React from "react";
import StarRating from "./StarRating";

export default function App(props){
  return <StarRating totalStars={6} style={{backgroundColor: "lightblue"}} onDoubleClick={e => alert(`double click`)}/>
}
