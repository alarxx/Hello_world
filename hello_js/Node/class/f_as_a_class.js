function Range(from, to){
	this.from = from;
	this.to = to;
}

Range.prototype = {
	includes(x){
		return x > this.from && x < this.to;
	}
}

let r = new Range(1, 5);
let b = r.includes(2);
console.log(b);