import React from 'react';

import useInput from './useInput.js';


export default function AddColor({addNewColor = f=>f}){
  const [titleProps, resetTitle] = useInput('');
  const [colorProps, resetColor] = useInput('#000000');

  const onSubmit = (e) => {
    e.preventDefault();
    addNewColor(titleProps.value, colorProps.value);
    resetTitle();
    resetColor();
  }

  return (
    <form onSubmit={onSubmit}>
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
