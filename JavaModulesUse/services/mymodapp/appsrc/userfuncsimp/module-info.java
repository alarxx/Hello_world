module userfuncsimp {
	requires userfuncs;

	provides userfuncs.binaryfuncs.BinaryFuncProvider with 
		userfuncsimp.binaryfuncsimp.AbsPlusProvider,
		userfuncsimp.binaryfuncsimp.AbsMinusProvider;
}