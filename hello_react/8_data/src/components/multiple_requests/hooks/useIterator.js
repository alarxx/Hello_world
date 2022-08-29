import React, {useState, useEffect, useCallback, useMemo} from 'react';

export default function useIterator(items=[], initialIndex=0){
  if(!items) return;
  const [i, setIndex] = useState(initialIndex);

  const prev = useCallback(() => {
    if(items.length === 0) return setIndex(0);
    if(i === 0) return setIndex(items.length - 1);
    setIndex(i - 1);
  }, [i]);

  const next = useCallback(() => {
    if(items.length === 0) return setIndex(0);
    if(i === items.length-1) return setIndex(0);
    setIndex(i + 1);
  }, [i]);

  const item = useMemo(()=>(items?items[i]:null), [i]);

  useEffect(()=>{
    console.log(i);
  }, [i]);

  return [item, prev, next]
}
