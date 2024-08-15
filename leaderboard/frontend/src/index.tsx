import React from 'react';
import ReactDOM from 'react-dom/client';
import 'react-tooltip/dist/react-tooltip.css'
import './index.css';
import {AppView} from "./component/app";


const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);
root.render(
  <React.StrictMode>
    <AppView />
  </React.StrictMode>
);
