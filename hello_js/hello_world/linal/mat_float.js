/** 
 * Типизированный массив (Float32Array)
 */
module.exports = class MatFloat {
	#mat;
	#rows;
	#cols;

	constructor(rows, cols){
		this.#rows = rows;
		this.#cols = cols;
		this.#mat = this.#createMat(rows, cols);
	}

	#createMat(rows, cols){
		let mat = new Float32Array(rows * cols).fill(0);
		return mat;
	}

	get(row, col){
		return this.#mat[col + row * this.#rows];
	}

	set(val, row, col){
		this.#mat[col + row * this.#rows] = val;
	}

	get getmat(){
		return this.#mat;
	}

	set setmat(mat){
		this.#mat = mat;
	}
}