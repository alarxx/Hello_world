import React, {createContext, useState, useContext} from 'react';
import {v4} from 'uuid'

import colorData from '../../assets/color-data.json';


const ColorContext = createContext();

export const useColors = () => useContext(ColorContext);

export function ColorProvider({ children }){
  const [colors, setColors] = useState(colorData);

  const addColor = (title, color) =>
    setColors(
      [
        ...colors,
        {
          id: v4(),
          rating: 0,
          title,
          color
        }
      ]
    );

  const rateColor = (id, rating) =>
    setColors(
      colors.map(color => color.id === id ? {...color, rating} : color)
    );

  const removeColor = id => setColors(colors.filter(color => id !== color.id))


  return (
    <ColorContext.Provider value={{colors, addColor, rateColor, removeColor}}>
      {children}
    </ColorContext.Provider>
  );
}
