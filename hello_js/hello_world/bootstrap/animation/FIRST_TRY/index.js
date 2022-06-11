const ob = document.querySelector("#ob");
console.log(document.documentElement.scrollWidth, document.documentElement.scrollHeight);
window.addEventListener("click", (e) => {
	console.log(e.clientX, e.clientY);
	
	let width = document.documentElement.scrollWidth;
	let height = document.documentElement.scrollHeight;


	ob.style.width = `${width / 2}px`;
	ob.style.height = `${height / 2}px`;

	console.log(ob.clientWidth, width /2)
	let x = e.clientX - ob.clientWidth / 2;
	let y = e.clientY - ob.clientHeight / 2;
	ob.style.left = `${x}px`;
	ob.style.top = `${y}px`;
});