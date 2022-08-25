import React, {useContext} from 'react';

import Color from './Color.js';
import {useColors} from "./color-hooks.js"

export default function ColorList(props){
  const {colors} = useColors();
  
  if(colors.length < 1)
    return <div>No Colors Listed. (Add a color)</div>;

  return (
    <div className="color-list">
      {colors.map(color => <Color key={color.id} {...color}/>)}
    </div>
  );
}
