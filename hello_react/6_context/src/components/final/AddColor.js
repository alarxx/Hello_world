import React from 'react';

import {useColors} from './color-hooks.js';
import useInput from './useInput.js';

export default function AddColor(){
  const [titleProps, resetTitle] = useInput('');
  const [colorProps, resetColor] = useInput('#000000');

  const {addColor} = useColors();

  const onSubmit = (e) => {
    e.preventDefault();
    addColor(titleProps.value, colorProps.value);
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
