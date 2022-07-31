//Неизменяемость аргументов функций
const color1 = {
	title: "lawn",
	color: "red",
	rating: 0
};

function rateColor(color, rating){
	color.rating = rating;
	return color; // Возвращает тот же самый объект
}

function rateColorCopy(color, rating){
	return Object.assign({}, color, {rating: rating});
}

// Но на этот раз напишем с помощью стрелочной функции
const rateColorCopySpread = (color, rating) => ({ 
		...color, 
		rating // same as {rating: rating}
	})


const color2 = rateColorCopySpread(color1, 5);

// console.log('color1.rating', color1.rating);
// console.log('color2.rating', color2.rating);



// Неизменяемость в массивах
const colors = [{title: "red"}, {title: "green"}];

//Изменит
const addColor2Initial = (colors, title) => {
	colors.push({title});
	return colors; // Зачем, если это тот же самый объект
};
//Не изменит
const addColor2Copy = (colors, title) => colors.concat({title});
const addColor2CopySpread = (colors, title) => [...colors, {title}]

// console.log(addColor2Initial(colors, "blue"));
// console.log(addColor2CopySpread(colors, "yellow"));
// console.log(...colors);