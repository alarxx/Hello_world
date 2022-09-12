import React from "react";
import {Routes, Route, Switch} from 'react-router-dom';

import {Home, About, Events, Contact, Products} from './basic_routes/pages';

export default function App(){
	return (
				<Routes>
					<Route path="/" element={<Home/>}/>
					<Route path="about" element={<About/>}/>
					<Route path="/events" element={<Events/>}/>
					<Route path="/contact" element={<Contact/>}/>
					<Route path="/products" element={<Products/>}/>
				</Routes>
	);
}
