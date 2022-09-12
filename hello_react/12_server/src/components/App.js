import React, {useState} from 'react';


export default function App(){
    const [count, setCount] = useState(0);

    const increment = ()=>setCount(count+1);
    const decrement = ()=>setCount(count-1);

    return(
        <>  
            <h1>Home</h1>
            <p>{count}</p>
            <button onClick={increment}>Increment</button>
            <button onClick={decrement}>Decrement</button>
        </>
    );
}