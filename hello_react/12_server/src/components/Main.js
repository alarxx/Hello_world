import React from "react";
import {Routes, Route} from 'react-router-dom';

import {Home, About, Events, Contact, Products} from './pages';

export default function Main(){
	return (
				<Routes>
					<Route path="/" element={<Home/>}/>
					<Route path="/about" element={<About/>}/>
					<Route path="/events" element={<Events/>}/>
					<Route path="/contact" element={<Contact/>}/>
					<Route path="/products" element={<Products/>}/>
				</Routes>
	);
}
