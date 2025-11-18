import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import '../assets/css/navbar.css';
import logo from '../assets/logo.png';

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <nav className={`navbar ${scrolled ? 'scrolled' : ''}`}>
      <div className="navbar-container">
        <Link to="/" className="navbar-logo">
          <img 
            src={logo} 
            alt="Decompressed Logo"
            style={{ height: '32px', width: 'auto' }}
          />
          <span>Decompressed</span>
        </Link>
        <div className="navbar-links">
          <Link to="/" className="navbar-link">Home</Link>
          <Link to="/demo" className="navbar-link">Demo</Link>
          <a 
            href="https://github.com/Dev-ZC/Decompressed" 
            className="navbar-link"
            target="_blank"
            rel="noopener noreferrer"
          >
            <i className="fab fa-github"></i> GitHub
          </a>
        </div>
      </div>
    </nav>
  );
}
