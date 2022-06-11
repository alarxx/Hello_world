module.exports = class Mat {
	// Можно использовать типизированный массив (Float32Array)
	#mat;
	#rows;
	#cols;
	
	constructor(rows, cols){
		this.#rows = rows;
		this.#cols = cols;
		this.#mat = this.#createMat(rows, cols);
	}

	#createMat(rows, cols){
		let mat = []
		for(let r=0; r<rows; r++){
			mat[r] = [];
			for(let c=0; c<cols; c++)
				mat[r][c] = 0;
		}
		return mat;
	};

	//returns new Mat
	copy(){
		let copyMat = new Mat(this.#rows, this.#cols);
		for(let r=0; r<this.#rows; r++)
			for(let c=0; c<this.#cols; c++)
				copyMat.set(this.#mat[r][c], r, c);
			
		return copyMat;
	}

	//не живая
	submat(s_row, s_col, rows, cols){
		let m = new Mat(rows, cols);
		for(let r=0; r<rows; r++)
			for(let c=0; c<cols; c++)
				m.set(this.get(s_row + r, s_col + c), r, c);
		return m;
	}

	get(row, col){
		return this.#mat[row][col];
	}

	set(val, row, col){
		this.#mat[row][col] = val;
	}

	row(){
		return this.#rows;
	}

	col(){
		return this.#cols;
	}

	get getmat(){
		return this.#mat;
	}

	set setmat(mat){
		this.#mat = mat;
	}

	toString(name){
		let str = `---${name}---\n`;

		for(let r=0; r<this.row(); r++){
			str += ` | `
			for(let c=0; c<this.col(); c++)
				str += `${this.getmat[r][c]} `;
			str += `|${r==this.row()-1?'':'\n'}`;
		}
		
		return str;
	}
}