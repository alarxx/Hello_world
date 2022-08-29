/**
  Для очень больших списков,
  когда инфа рендерится частями, а не вся сразу
*/

import React, {useState, useEffect} from 'react';
import {FixedSizeList} from 'react-window';


//Должно быть много разных людей или любая "бесконечная" инфа
const bigList = [...new Array(5000)].map(()=>({
  name: "Alar",
  email: "alar@gmail.com",
  avatar: "https://avatars.githubusercontent.com/u/72505048?s=400&u=f9d0582e1b2a7f64798d3c4251d5415371db5a9a&v=4"
}));


export default function Main(){
  const renderRow = ({index, style}) => (
    <div style={{...style, ...{display: "flex"}}}>
      <img src={bigList[index].avatar} alt={bigList[index].name} width={50}/>
      <p>
        {bigList[index].name} - {bigList[index].email}
      </p>
    </div>
  );
  return (
    <FixedSizeList
      height={500}
      width={500}
      itemCount={bigList.length}
      itemSize={50}
    >
      {renderRow}
    </FixedSizeList>
  );
}
