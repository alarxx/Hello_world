function wait(duration){
	return new Promise((resolve, reject)=>{
		if(duration < 0){
			reject(new Error("Time travel not yet implemented"));
		}
		else{
			setTimeout(resolve, duration);
			return "lol"
		}
	});
}

wait(1000).then(res => console.log(res));