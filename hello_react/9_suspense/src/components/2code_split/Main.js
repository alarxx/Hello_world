/**
  Решение с handler компонентами более декларативное, чем на основе useFetch
*/

import React, { useState, lazy, Suspense } from 'react';
import ClipLoader from "react-spinners/ClipLoader";

import Agreement from './Agreement.js';
import ErrorBoundary from '../1error_bounds/ErrorBoundary.js';
import ThrowError from '../1error_bounds/ThrowError.js';
import Status from './Status.js'

const BigApp = lazy(() => import('./BigApp.js'));


export default function Main(){
  const [agree, setAgree] = useState(false);

  if(!agree)
    return (<Agreement onAgree={setAgree}/>);

  return (
    <Suspense fallback={<ClipLoader/>}>
        <Status />
    </Suspense>
  )
}
