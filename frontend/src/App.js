// src/App.js
import React from 'react';
import { BrowserRouter as Router, Route, Switch } from 'react-router-dom';
import HateSpeechDetector from './components/HateSpeechDetector';
import NavBar from './components/NavBar';
import Dashboard from './components/Dashboard';
import './App.css';

function App() {
  return (
    <Router>
      <div className="app">
        <NavBar />
        <Switch>
          <Route exact path="/" component={HateSpeechDetector} />
          <Route path="/Dashboard" component={Dashboard} />
        </Switch>
      </div>
    </Router>
  );
}

export default App;
