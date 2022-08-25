import React, {useState} from "react";
import useInput from "./useInput.js"

const black = "#000000";

const defaultOnNewColor = (t, c)=>{console.log("my controlled",t, c);};


export default function MyForm({onNewColor = defaultOnNewColor}){
  const [titleProps, resetTitle] = useInput("");
  const [colorProps, resetColor] = useInput(black);
  const submit = e => {
    e.preventDefault();
    onNewColor(titleProps.value, colorProps.value);
    resetTitle();
    resetColor();
  }

  return (
    <form onSubmit={submit}>
      <input
        {...titleProps}
        type="text"
        placeholder="color title..."
        required
      />
      <input
        {...colorProps}
        type="color"
        required
      />
      <button>Add</button>
    </form>
  );
}