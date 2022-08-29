import React, {useState, useEffect} from 'react';


function List({data = [], renderEmpty, renderItem}){
  if(data.length === 0)
    return renderEmpty;
  else return (
    <ul>{
        data.map((item, i)=>
          <li key={i}>{renderItem(item)}</li>
        )
    }</ul>
  );
}


const bigList = [...new Array(50000)].map(()=>({
  name: "Alar",
  email: "alar@gmail.com",
  avatar: "https://avatars.githubusercontent.com/u/72505048?s=400&u=f9d0582e1b2a7f64798d3c4251d5415371db5a9a&v=4"
}));


export default function Main(){
  const renderItem = item => (
    <div style={{display:"flex"}}>
      <img src={item.avatar} alt={item.name} width={50} />
      <p>
        {item.name} - {item.email}
      </p>
    </div>
  );
  return (
    <List
      data={bigList}
      renderItem={renderItem}
    />
  );
}
