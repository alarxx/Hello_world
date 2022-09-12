import React, {Component} from 'react';
import ReactDOM from 'react-dom/client';

// import 'bootstrap/dist/css/bootstrap.min.css';
import {BrowserRouter} from 'react-router-dom';

import Main from "./components/Main.js"

const root = ReactDOM.createRoot(document.getElementById("root"));

root.render(
  <BrowserRouter>
    <Main/>
  </BrowserRouter>
);
