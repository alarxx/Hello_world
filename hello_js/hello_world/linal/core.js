const Mat = require('./mat.js');
module.exports = class Core {

	static _add(mat, val){
		if(val instanceof Mat)
			for(let r=0; r<mat.row(); r++)
				for(let c=0; c<mat.col(); c++)
					mat.getmat[r][c] += val.getmat[r][c];
		else if(typeof val === 'number')
			for(let r=0; r<mat.row(); r++)
				for(let c=0; c<mat.col(); c++)
					mat.getmat[r][c] += val;
	}

	static add(mat, val){
		let result = mat.copy();
		if(val instanceof Mat)
			for(let r=0; r<mat.row(); r++)
				for(let c=0; c<mat.col(); c++)
					result.getmat[r][c] += val.getmat[r][c];
		else if(typeof val === 'number')
			for(let r=0; r<mat.row(); r++)
				for(let c=0; c<mat.col(); c++)
					result.getmat[r][c] += val;
		return result;
	}

}