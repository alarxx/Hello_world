import React from 'react';
import ReactDOM from 'react-dom';

import {BrowserRouter, Routes, Route} from 'react-router-dom';

import {Home, About, Events, Contact, Products} from './components/pages.js';
import Counter from './components/Counter.js';

ReactDOM.render(
  <BrowserRouter>
    <Routes>
          <Route path="/" element={<Home/>}/>
          <Route path="/about" element={<About/>}/>
          <Route path="/events" element={<Events/>}/>
          <Route path="/contact" element={<Contact/>}/>
          <Route path="/products" element={<Products/>}/>
          <Route path="/counter" element={<Counter/>}/>
    </Routes>
  </BrowserRouter>
, document.getElementById("root"));


