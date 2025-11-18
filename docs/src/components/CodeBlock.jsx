import React, { useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import './CodeBlock.css';

export default function CodeBlock({ code, language = 'python', title, icon = 'fas fa-code' }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  const customStyle = {
    margin: 0,
    padding: '1.5rem',
    background: 'transparent',
    fontSize: '0.95rem',
    borderRadius: 0,
  };

  return (
    <div className="code-block-container">
      <div className="code-header">
        <div className="code-header-left">
          <i className={icon}></i>
          <span>{title}</span>
        </div>
        <button 
          className={`copy-button ${copied ? 'copied' : ''}`}
          onClick={handleCopy}
          title="Copy to clipboard"
        >
          {copied ? (
            <>
              <i className="fas fa-check"></i>
              <span>Copied!</span>
            </>
          ) : (
            <>
              <i className="far fa-copy"></i>
              <span>Copy</span>
            </>
          )}
        </button>
      </div>
      <SyntaxHighlighter
        language={language}
        style={vscDarkPlus}
        customStyle={customStyle}
        showLineNumbers={false}
        wrapLongLines={true}
      >
        {code}
      </SyntaxHighlighter>
    </div>
  );
}
