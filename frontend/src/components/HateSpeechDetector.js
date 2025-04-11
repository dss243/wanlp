import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

function HateSpeechDetector() {
  const [inputText, setInputText] = useState('');
  const [results, setResults] = useState({
    hate_speech: null,
    topic: null,
    timestamp: null
  });
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);
  const [redditAuth, setRedditAuth] = useState({
    isAuthenticated: false,
    user: null,
    token: null
  });
  const [redditAnalysis, setRedditAnalysis] = useState(null);
  const [analyzingReddit, setAnalyzingReddit] = useState(false);

  const API_URL = 'http://localhost:8000';
  
  // Handle manual text analysis (existing functionality)
  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    
    try {
      const response = await axios.post(`${API_URL}/predict/`, 
        { text: inputText },
        { headers: { 'Content-Type': 'application/json' } }
      );
      
      setResults({
        hate_speech: response.data.hate_speech,
        topic: response.data.topic,
        timestamp: response.data.timestamp
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
      handleApiError(err);
    } finally {
      setLoading(false);
    }
  };

  // Handle Reddit OAuth login
  const handleRedditLogin = () => {
    window.location.href = `${API_URL}/login/reddit`;
  };

  // Check for auth callback on initial load
  useEffect(() => {
    const urlParams = new URLSearchParams(window.location.search);
    const code = urlParams.get('code');
    
    if (code) {
      exchangeRedditCode(code);
    }
  }, []);

  // Exchange authorization code for token
  const exchangeRedditCode = async (code) => {
    try {
      const response = await axios.get(`${API_URL}/auth/callback?code=${code}`);
      setRedditAuth({
        isAuthenticated: true,
        token: response.data.access_token
      });
      // Clean URL
      window.history.replaceState({}, document.title, window.location.pathname);
    } catch (err) {
      handleApiError(err);
    }
  };

  // Analyze user's Reddit content
  const analyzeRedditContent = async () => {
    setAnalyzingReddit(true);
    setError(null);
    
    try {
      const response = await axios.get(`${API_URL}/analyze-me/`, {
        headers: {
          'Authorization': `Bearer ${redditAuth.token}`
        }
      });
      
      setRedditAnalysis(response.data);
    } catch (err) {
      handleApiError(err);
    } finally {
      setAnalyzingReddit(false);
    }
  };

  const handleApiError = (err) => {
    console.error("API Error:", err);
    const detail = err.response?.data?.detail || err.message;
    setError(detail || 'حدث خطأ أثناء الاتصال بالخادم');
  };

  const clearAll = () => {
    setInputText('');
    setResults({
      hate_speech: null,
      topic: null,
      timestamp: null
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
        {/* Manual Analysis Section */}
        <section className="input-section">
          <h2>تحليل نص يدوي</h2>
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

        {/* Reddit Analysis Section */}
        <section className="reddit-section">
          <h2>تحليل محتوى ريديت</h2>
          
          {!redditAuth.isAuthenticated ? (
            <div className="reddit-auth">
              <p>للوصول إلى تحليل مشاركاتك على ريديت، يرجى تسجيل الدخول:</p>
              <button 
                className="reddit-login-btn"
                onClick={handleRedditLogin}
              >
                تسجيل الدخول عبر ريديت
              </button>
            </div>
          ) : (
            <div className="reddit-analysis">
              <button
                className="analyze-reddit-btn"
                onClick={analyzeRedditContent}
                disabled={analyzingReddit}
              >
                {analyzingReddit ? 'جارٍ تحليل محتواك...' : 'تحليل مشاركاتي الأخيرة'}
              </button>
              
              {redditAnalysis && (
                <div className="reddit-results">
                  <h3>ملخص تحليل ريديت</h3>
                  <div className="stats-grid">
                    <div className="stat-card">
                      <span className="stat-value">{redditAnalysis.total_posts}</span>
                      <span className="stat-label">منشورات</span>
                    </div>
                    <div className="stat-card">
                      <span className="stat-value">{redditAnalysis.total_comments}</span>
                      <span className="stat-label">تعليقات</span>
                    </div>
                    <div className="stat-card">
                      <span className="stat-value">{redditAnalysis.hate_speech_percentage.toFixed(1)}%</span>
                      <span className="stat-label">خطاب كراهية</span>
                    </div>
                  </div>
                  
                  <h4>توزيع المواضيع</h4>
                  <div className="topics-distribution">
                    {Object.entries(redditAnalysis.topics_distribution).map(([topic, count]) => (
                      <div key={topic} className="topic-item">
                        <span className="topic-name">{topic}</span>
                        <span className="topic-count">{count}</span>
                      </div>
                    ))}
                  </div>
                  
                  <h4>نماذج من النتائج</h4>
                  <div className="sample-results">
                    {redditAnalysis.sample_results.map((result, index) => (
                      <div key={index} className={`sample-result ${result.hate_speech === 'Hate Speech' ? 'hate' : 'clean'}`}>
                        <p className="sample-text">
                          {result.text.length > 100 ? `${result.text.substring(0, 100)}...` : result.text}
                        </p>
                        <div className="sample-meta">
                          <span className={`sample-label ${result.hate_speech === 'Hate Speech' ? 'hate' : 'clean'}`}>
                            {result.hate_speech === 'Hate Speech' ? 'خطاب كراهية' : 'نص آمن'}
                          </span>
                          <span className="sample-topic">{result.topic}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </section>

        {/* Error Display */}
        {error && (
          <div className="error-message">
            <p>⚠️ خطأ: {error}</p>
          </div>
        )}

        {/* Manual Analysis Results */}
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
              
              {results.timestamp && (
                <div className="result-item">
                  <span className="result-label">وقت التحليل:</span>
                  <span className="result-value">
                    {new Date(results.timestamp).toLocaleString()}
                  </span>
                </div>
              )}
            </div>
          </section>
        )}

        {/* History Section */}
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

export default HateSpeechDetector;
