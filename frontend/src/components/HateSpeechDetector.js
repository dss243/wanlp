import React, { useState } from 'react';
import axios from 'axios';
import './App.css'; // قم بإنشاء هذا الملف لتنسيق الواجهة

function App() {
  const [inputText, setInputText] = useState('');
  const [results, setResults] = useState({
    hate_speech: null,
    topic: null,
    confidence: null
  });
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);
  
  const API_URL = 'http://localhost:8000/predict/';

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    
    try {
      const response = await axios.post(API_URL, 
        { text: inputText },
        {
          headers: {
            'Content-Type': 'application/json',
          }
        }
      );
      
      setResults({
        hate_speech: response.data.hate_speech,
        topic: response.data.topic,
        confidence: response.data.confidence || "N/A"
      });
      
      setHistory(prev => [
        {
          text: inputText,
          result: response.data,
          timestamp: new Date().toLocaleString()
        },
        ...prev.slice(0, 4)
      ]);
      
    } catch (err) {
      console.error("API Error:", err);
      const detail = err.response?.data?.detail;

      if (Array.isArray(detail)) {
        setError(detail.map(d => d.msg).join(', '));
      } else if (typeof detail === 'object' && detail !== null) {
        setError(detail.msg || JSON.stringify(detail));
      } else {
        setError(detail || err.message || 'حدث خطأ أثناء الاتصال بالخادم');
      }
    } finally {
      setLoading(false);
    }
  };

  const clearAll = () => {
    setInputText('');
    setResults({
      hate_speech: null,
      topic: null,
      confidence: null
    });
    setError(null);
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>محلل المحتوى العربي</h1>
        <p>تحليل النصوص العربية لاكتشاف خطاب الكراهية وتصنيف المواضيع</p>
      </header>

      <main className="main-content">
        <section className="input-section">
          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label htmlFor="arabic-text">أدخل نصًا باللغة العربية:</label>
              <textarea
                id="arabic-text"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="اكتب أو الصق نصًا باللغة العربية هنا..."
                rows="5"
                required
              />
            </div>
            
            <div className="button-group">
              <button 
                type="submit" 
                className="analyze-btn"
                disabled={loading || !inputText.trim()}
              >
                {loading ? (
                  <>
                    <span className="spinner"></span> جارٍ التحليل...
                  </>
                ) : 'تحليل النص'}
              </button>
              
              <button 
                type="button" 
                className="clear-btn"
                onClick={clearAll}
                disabled={!inputText && !results.hate_speech}
              >
                مسح
              </button>
            </div>
          </form>
        </section>

        {error && (
          <div className="error-message">
            <p>⚠️ خطأ: {error}</p>
          </div>
        )}

        {(results.hate_speech || results.topic) && !error && (
          <section className="results-section">
            <h2>نتائج التحليل</h2>
            
            <div className={`result-card ${results.hate_speech === 'Hate Speech' ? 'hate' : 'clean'}`}>
              <div className="result-item">
                <span className="result-label">خطاب كراهية:</span>
                <span className="result-value">
                  {results.hate_speech === 'Hate Speech' ? 'نعم' : 'لا'}
                  {results.hate_speech === 'Hate Speech' ? (
                    <span className="warning-icon">⚠️</span>
                  ) : (
                    <span className="safe-icon">✓</span>
                  )}
                </span>
              </div>
              
              <div className="result-item">
                <span className="result-label">الموضوع:</span>
                <span className="result-value">{results.topic}</span>
              </div>
              
              {results.confidence && (
                <div className="result-item">
                  <span className="result-label">نسبة الثقة:</span>
                  <span className="result-value">{results.confidence}</span>
                </div>
              )}
            </div>
          </section>
        )}

        {history.length > 0 && (
          <section className="history-section">
            <h3>التحليلات الأخيرة</h3>
            <ul className="history-list">
              {history.map((item, index) => (
                <li key={index} className="history-item">
                  <div className="history-text">
                    <p>"{item.text.length > 50 ? `${item.text.substring(0, 50)}...` : item.text}"</p>
                    <small>{item.timestamp}</small>
                  </div>
                  <div className={`history-result ${item.result.hate_speech === 'Hate Speech' ? 'hate' : 'clean'}`}>
                    {item.result.hate_speech === 'Hate Speech' ? 'خطاب كراهية' : 'نص آمن'}
                  </div>
                </li>
              ))}
            </ul>
          </section>
        )}
      </main>

      <footer className="app-footer">
        <p>© {new Date().getFullYear()} محلل المحتوى العربي</p>
      </footer>
    </div>
  );
}

export default App;
