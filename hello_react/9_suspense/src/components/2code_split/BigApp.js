import React from 'react';

import ErrorBoundary from '../1error_bounds/ErrorBoundary.js'
import ThrowError from '../1error_bounds/ThrowError.js'
import Status from './Status.js'

// Представим, что это очень большой модул

export default function BigApp(){
  return (
    <>
      <h1>Very big App</h1>
      <Status />
    </>
  );
}
