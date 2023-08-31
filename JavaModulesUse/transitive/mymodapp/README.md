```
module appsupport {
	exports appsupport.supportfuncs;	
}

module appfuncs {
	exports appfuncs.simplefuncs;	
	requires transitive appsupport;	
}

module appstart {
	requires appfuncs;	
	// requires appsupport;	
}
```

appfuncs requires appsupport  
appstart requires appfuncs 

So appstart indirect dependent and automatically requires appsupport module.