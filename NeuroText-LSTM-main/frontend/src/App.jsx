import React, { useState, useEffect, useRef } from 'react';
import './index.css';

const App = () => {
  const [prompt, setPrompt] = useState('The project aim is');
  const [length, setLength] = useState(100);
  const [temperature, setTemperature] = useState(0.5);
  const [topK, setTopK] = useState(5);
  const [generatedText, setGeneratedText] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [status, setStatus] = useState('idle');
  const [error, setError] = useState('');
  
  const resultRef = useRef(null);

  useEffect(() => {
    checkStatus();
  }, []);

  const checkStatus = async () => {
    try {
      const response = await fetch('http://localhost:5000/status');
      const data = await response.json();
      setStatus(data.status);
    } catch (err) {
      console.error('Backend is not running', err);
      setStatus('disconnected');
    }
  };

  const handleGenerate = async () => {
    if (!prompt) return;
    setIsGenerating(true);
    setGeneratedText('');
    setError('');
    
    try {
      const response = await fetch('http://localhost:5000/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          prompt,
          length,
          temperature,
          top_k: topK
        }),
      });
      
      const data = await response.json();
      
      if (data.error) {
        setError(data.error);
        setIsGenerating(false);
        return;
      }

      // Simulate character-by-character typing effect
      let fullText = data.text;
      let currentIndex = 0;
      
      const timer = setInterval(() => {
        setGeneratedText(fullText.substring(0, currentIndex + 1));
        currentIndex++;
        
        if (currentIndex === fullText.length) {
          clearInterval(timer);
          setIsGenerating(false);
        }
        
        // Scroll to bottom
        if (resultRef.current) {
          resultRef.current.scrollTop = resultRef.current.scrollHeight;
        }
      }, 50);

    } catch (err) {
      setError('Could not connect to the backend server. Make sure it is running on port 5000.');
      setIsGenerating(false);
    }
  };

  return (
    <div className="app-container">
      <header>
        {status === 'ready' && (
          <div className="status-badge">
            <span className="badge-dot"></span>
            Model Ready
          </div>
        )}
        {status === 'need_training' && (
          <div className="status-badge training">
            <span className="badge-dot"></span>
            Needs Training
          </div>
        )}
        {status === 'disconnected' && (
          <div className="status-badge error">
            <span className="badge-dot"></span>
            Backend Disconnected
          </div>
        )}
        
        <h1>NeuroText LSTM</h1>
        <p className="subtitle">Interactive AI Text Generation powered by PyTorch</p>
      </header>

      <main className="controls">
        <div className="input-group">
          <label htmlFor="prompt">Seed Text (Prompt)</label>
          <input 
            id="prompt"
            className="prompt-input"
            type="text" 
            value={prompt} 
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Type something to start..."
            disabled={isGenerating}
          />
        </div>

        <div className="settings-row">
          <div className="slider-container">
            <label>Length: {length} chars</label>
            <input 
              type="range" 
              min="20" 
              max="1000" 
              value={length} 
              onChange={(e) => setLength(parseInt(e.target.value))}
              disabled={isGenerating}
            />
          </div>
          <div className="slider-container">
            <label>Top K: {topK}</label>
            <input 
              type="range" 
              min="1" 
              max="20" 
              value={topK} 
              onChange={(e) => setTopK(parseInt(e.target.value))}
              disabled={isGenerating}
            />
          </div>
        </div>

        <button 
          className="generate-btn" 
          onClick={handleGenerate}
          disabled={isGenerating || status === 'disconnected'}
        >
          {isGenerating ? 'Generating...' : 'Initiate Text Sequence'}
        </button>

        <div className="result-area" ref={resultRef}>
          {generatedText ? (
            <>
              {generatedText}
              {isGenerating && <span className="cursor"></span>}
            </>
          ) : (
            !isGenerating && (
              <div className="empty-state">
                {error ? (
                  <span style={{color: 'var(--error)'}}>{error}</span>
                ) : (
                  'Generated text will appear here...'
                )
                }
              </div>
            )
          )}
          {isGenerating && !generatedText && (
            <div className="generating-text">Architecting thoughts...</div>
          )}
        </div>
      </main>

      <footer style={{marginTop: '2rem', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.8rem'}}>
        Built with PyTorch & React • Character-level LSTM Architecture
      </footer>
    </div>
  );
};

export default App;
