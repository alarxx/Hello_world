import React, {useState} from "react";


function useInput(initialValue) {
  const [value, setValue] = useState(initialValue);
  return [
    {value, onChange: e => setValue(e.target.value)},
    () => setValue(initialValue)
  ];
}


export default function SearchForm({onSearch = f=>f}){
  const [titleProps, resetTitle] = useInput("");

  const submit = e => {
    e.preventDefault();
    onSearch(titleProps.value);
    resetTitle();
  }

  return (
    <form onSubmit={submit}>
      <input
        value={titleProps.value}
        onChange={titleProps.onChange}
        type="text"
        placeholder="github user login"
        required
      />
      <button>Search</button>
    </form>
  );
}
