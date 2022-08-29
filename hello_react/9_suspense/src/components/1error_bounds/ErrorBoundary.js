import React, {Component} from 'react';


function DefaultErrorScreen({ error }) {
  const errorCss = {
    backgroundColor: '#efacac',
    border: 'double 4px darkred',
    color: 'darkred',
    padding: '1em',
  };

  return (
    <div style={errorCss}>
      <h3>We are sorry... something went wrong</h3>
      <p>We cannot process your request at this moment.</p>
      <p>ERROR: {error.message}</p>
    </div>
  );
}


// Многоразовый комнент error handler
export default class ErrorBoundary extends Component{
  constructor(props) {
    super(props);
    this.state = {error: null};
  }

  static getDerivedStateFromError(error){
    return { error };
  }

  render(){
    const {error} = this.state;
    const {children, ErrorScreen} = this.props;

    if(error && !ErrorScreen)
      return <DefaultErrorScreen error={error}/>
    else if(error)
      return <ErrorScreen error={error}/>
    else
      return children;
  }
}
