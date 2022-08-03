import React, {useState} from "react";
import {FaStar} from "react-icons/fa";

const Star = ({selected = false, onSelect = (f) => f}) => (<FaStar color={selected?"red":"grey"} onClick={onSelect}/>);

const createArray = (len) => [...new Array(len)];


export default function App({style={}, totalStars=5, ...props}){
	const [selectedStars, setSelectedStars] = useState(0); //Второй элемент из useState - setter для первого
	return (
		<>
			<h1>Hello, React!</h1>
			<div style={{padding: "5px", ...style}} {...props}>
				{createArray(totalStars)
					.map((n, i) => ( // n - null, мы заменяем его <Star/> компонентом
						<Star
							key={i}
							selected={selectedStars > i}
							onSelect={() => setSelectedStars(i+1)}
						/>
					)
				)}
			</div>
			<p>{selectedStars} of {totalStars}</p>
		</>
	);
}
