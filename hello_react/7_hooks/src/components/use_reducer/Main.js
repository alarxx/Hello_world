import React, {useState, useReducer} from 'react';

const firstUser = {
  id: "01",
  firstName: "Bill",
  lastName: "Willson",
  email: "bill.willson@gmail.com",
  city: "Astana",
  admin: false
};

export default function Main(){
  const [user, setUser] = useReducer((user, newDetails)=>({...user, ...newDetails}), firstUser);

  return (
    <div>
        <h1>{user.firstName} {user.lastName} - {user.admin ? "Admin" : "User"}</h1>
        <p>Email: {user.email}</p>
        <p>Location: {user.city}</p>
        <button onClick={()=>{
          setUser({admin: true});
        }}>Make Admin</button>
    </div>
  );
}
