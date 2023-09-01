package userfuncsimp.binaryfuncsimp;

import userfuncs.binaryfuncs.*;

public class AbsMinusProvider implements BinaryFuncProvider {
	@Override
	public BinaryFunc get(){
		return new AbsMinus();
	}
}