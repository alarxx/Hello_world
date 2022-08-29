import React from 'react';

import ErrorBoundary from './ErrorBoundary.js'
import ErrorScreen from './ErrorScreen.js'
import ThrowError from './ThrowError.js'


export default function Main() {
  return (
    <>
      <ErrorBoundary>
        <h1>App</h1>
        <ThrowError />
      </ErrorBoundary>
    </>
  );
}
