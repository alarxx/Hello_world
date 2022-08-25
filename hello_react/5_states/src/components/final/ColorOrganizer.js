import React, {useState} from 'react';
import {v4} from 'uuid'

import colorData from '../../assets/color-data.json'
import ColorList from './ColorList.js'
import AddColor from './AddColor.js'

export default function ColorRating(){
  const [colors, setColors] = useState(colorData);
  return (
    <div>
      <h1>Color Organizer</h1>
      <h3>List of Colors</h3>
      <ColorList
        colors={colors}
        onRemoveColor={(id) => {
          const newColors = colors.filter(color => id !== color.id);
          setColors(newColors);
        }}
        onRateColor={(id, rating)=>{
          const newColors = colors.map(color => (id === color.id ? {...color, rating} : color));
          setColors(newColors);
        }}
      />
      <h3>Add User Color</h3>
      <AddColor addNewColor={
        (title, color)=>{
          const newColors = [...colors,
            {
              id:v4(),
              rating: 0,
              title,
              color
            }
          ];
          setColors(newColors);
        }
      }/>
    </div>
  );
}
