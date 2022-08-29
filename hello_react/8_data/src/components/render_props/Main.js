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

const tahoe_peaks = [
  {name: "Freel Peak", elevation: 10891},
  {name: "Monument Peak", elevation: 10067},
  {name: "Pyramid Peak", elevation: 9983},
  {name: "Mt. Tallac", elevation: 9735},
]

export default function Main(){
  return (
    <List
      data={tahoe_peaks}
      renderEmpty={<p>This list is empty</p>}
      renderItem={item => (`${item.name} - ${item.elevation}`)}
    />
  );
}
