// src/components/NavBar.js
import React from 'react';

import { Link } from 'react-router-dom';
import './App.css';
function NavBar() {
  return (
    <nav className="navbar">
      <div className="navbar-container">
        <Link to="/" className="navbar-logo">
          <img src="/logiya.png" alt="Company Logo" className="logo-img" />
        </Link>
        <ul className="nav-menu">
          <li className="nav-item">
            <Link to="/" className="nav-links">تحليل نص يدوي</Link>
          </li>
          <li className="nav-item">
            <Link to="/Dashboard" className="nav-links">لوحة القيادة</Link>
          </li>
          <li className="nav-item">
            <button className="reddit-login-btn">
              تسجيل الدخول عبر ريديت
            </button>
          </li>
        </ul>
      </div>
    </nav>
  );
}

export default NavBar;