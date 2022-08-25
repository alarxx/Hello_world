import React, {createContext} from 'react';

import {ColorProvider} from './final/color-hooks.js'
import ColorOrganizer from './final/ColorOrganizer.js'

export default function App(){
  return (
    <ColorProvider>
      <ColorOrganizer/>
    </ColorProvider>
  );
}
