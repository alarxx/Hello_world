import React, {useRef} from "react";

const defaultOnNewColor = (t, c)=>{console.log(t, c);};

export default function App({onNewColor = defaultOnNewColor}){
  const txtTitle = useRef();
  const hexColor = useRef();

  const submit = e => {
    e.preventDefault();
    const title = txtTitle.current.value;
    const color = hexColor.current.value;
    onNewColor(title, color);
    txtTitle.current.value = "";
    hexColor.current.value = "";
  }

  return (
    <form onSubmit={submit}>
      <input ref={txtTitle} type="text" placeholder="color title..." required />
      <input ref={hexColor} type="color" required />
      <button>Add</button>
    </form>
  );
}
