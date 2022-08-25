import React, {useState} from 'react';

import ColorList from './ColorList.js'
import AddColor from './AddColor.js'

export default function ColorRating(){
  return (
    <>
      <ColorList />
      <AddColor />
    </>
  );
}
