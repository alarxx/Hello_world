import React, { useState } from "react";

//https://github.com/dustinspecker/awesome-eslint

function gnar() {
  const [v, sv] = useState();
  return v;
}

function Image() {
  console.log("lol");
  return <img src="/img.png" />;
}

export default function Fetch() {
  const [v, sv] = useState();
  return <h1>lol {v}</h1>;
}
