function fillGrid(id, n, dir, ext, links){
	const mygrid = document.getElementById(id);
	for(let i=0; i<n; i++){
		const mainEl = document.createElement("div");
		mainEl.setAttribute("class", "col-6 col-lg-3 col-md-4 col-sm-6 swipperCol hoverscale");

		const aElement = document.createElement("a");
		aElement.setAttribute("href", links[i]);

		const imgEl = document.createElement("img");
		imgEl.setAttribute("class", "img-fluid swipperImg hoverscale");
		imgEl.setAttribute("src", `${dir}/${i+1}.${ext}`);

		aElement.appendChild(imgEl);
		mainEl.appendChild(aElement);
		mygrid.appendChild(mainEl);
	}
}