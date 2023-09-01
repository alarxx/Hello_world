package userfuncsimp.binaryfuncsimp;

import userfuncs.binaryfuncs.*;

public class AbsPlusProvider implements BinaryFuncProvider {
	@Override
	public BinaryFunc get(){
		return new AbsPlus();
	}
}