import React from "react";

export default function App(props){
  return (
    <form>
      <input type="text" placeholder="color title..." required />
      <input type="color" required />
      <button>Add</button>
    </form>
  );
}
