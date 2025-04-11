import axios from 'axios';
import React, { useEffect, useState } from 'react';
import { useHistory } from 'react-router-dom';
import './App.css';

function Dashboard() {
  const [userData, setUserData] = useState(null);
  const [posts, setPosts] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [redditAuth, setRedditAuth] = useState({
    isAuthenticated: false,
    user: null,
    token: null
  });
  const navigate = useHistory();

  useEffect(() => {
    // Check URL for auth params first
    const params = new URLSearchParams(window.location.search);
    const access_token = params.get('access_token');

    if (access_token) {
      const authData = {
        isAuthenticated: true,
        token: access_token,
        user: {
          name: params.get('user_name'),
          id: params.get('user_id'),
          icon_img: params.get('icon_img')
        }
      };

      setRedditAuth(authData);
      localStorage.setItem('redditAuth', JSON.stringify(authData));
      window.history.replaceState({}, document.title, "/dashboard");
    } else {
      // Fallback to localStorage
      const storedAuth = localStorage.getItem('redditAuth');
      if (storedAuth) {
        setRedditAuth(JSON.parse(storedAuth));
      } else {
        navigate('/'); // Redirect to home if not authenticated
      }
    }
  }, [navigate]);

  useEffect(() => {
    if (redditAuth.isAuthenticated) {
      setLoading(true);
      // Simulate user data fetching
      setUserData({
        name: redditAuth.user.name,
        username: redditAuth.user.name,
        joinDate: new Date().toLocaleDateString('ar-AR', {
          year: 'numeric',
          month: 'long'
        }),
        postCount: 0,
        flaggedComments: 0
      });
      setLoading(false);
    }
  }, [redditAuth]);

  return (
    <div className="dashboard-container">
      {loading ? (
        <div className="loading">جار التحميل...</div>
      ) : error ? (
        <div className="error-message">⚠️ خطأ: {error}</div>
      ) : (
        <>
          <div className="user-profile">
            <div className="profile-header">
              <div className="user-info">
                {redditAuth.user?.icon_img && (
                  <img
                    src={redditAuth.user.icon_img}
                    alt="Profile"
                    className="profile-img"
                  />
                )}
                <div className="user-text">
                  <h2 className="username">@{userData?.username}</h2>
                  <p className="join-date">انضم في {userData?.joinDate}</p>
                </div>
              </div>
            </div>
            {/* You can continue your dashboard components here */}
            <div className="dashboard-content">
              <p>📊 لا توجد منشورات حتى الآن.</p>
              <p>📝 التعليقات التي تم الإبلاغ عنها: {userData?.flaggedComments}</p>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

export default Dashboard;