import React, {useState} from "react";
import ColorData from "../assets/color-data.json";
import ColorList from "./ColorList.js";

export default function ColorOrganizer(){
  const [colors, setColors] = useState(ColorData);
  return(
    <ColorList
      colors={colors}
      onRemoveColor={id => {
        const newColors = colors.filter(color => color.id !== id);
        setColors(newColors)
      }}
      onRateColor={(id, rating)=>{
        const newColors = colors.map(color => color.id === id ? {...color, rating} : color);
        setColors(newColors);
      }}
    />
  );
}
