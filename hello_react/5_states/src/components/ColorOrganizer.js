import React, {useState} from "react";
import ColorData from "../assets/color-data.json";
import ColorList from "./ColorList.js";

export default function ColorOrganizer(){
  const [colors] = useState(ColorData);
  return <ColorList colors={colors}/>
}
