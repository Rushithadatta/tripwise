import React, { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { getAuth, onAuthStateChanged, signOut } from 'firebase/auth'

export default function Navbar() {
  const [dark, setDark] = useState(() => {
    const saved = localStorage.getItem('darkMode')
    return saved === 'true'
  })
  const [user, setUser] = useState(null)
  const [showProfile, setShowProfile] = useState(false)

  useEffect(() => {
    if (dark) {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
    localStorage.setItem('darkMode', dark)
  }, [dark])

  useEffect(() => {
    const auth = getAuth()
    const unsubscribe = onAuthStateChanged(auth, (u) => setUser(u))
    return () => unsubscribe()
  }, [])

  function toggleDark() {
    setDark(d => !d)
  }

  function handleProfileClick() {
    setShowProfile((prev) => !prev)
  }

  function handleLogout() {
    const auth = getAuth()
    signOut(auth)
    setShowProfile(false)
  }

  return (
    <nav className={dark ? 'bg-gray-900 shadow-sm' : 'bg-white shadow-sm'}>
      <div className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
        <Link 
          to="/" 
          className={dark ? 'font-bold text-xl text-blue-400' : 'font-bold text-xl text-blue-600'}
        >
          TripWise
        </Link>

        <div className="space-x-4 flex items-center relative">
          <Link 
            to="/plan" 
            className={dark ? 'px-4 py-2 rounded-md hover:bg-gray-800 text-white' : 'px-4 py-2 rounded-md hover:bg-gray-100'}
          >
            Plan Trip
          </Link>

          {!user && (
            <Link 
              to="/login" 
              className={dark ? 'px-4 py-2 rounded-md bg-blue-700 text-white' : 'px-4 py-2 rounded-md bg-blue-600 text-white'}
            >
              Login
            </Link>
          )}

          {user && (
            <div className="relative inline-block">
              <img
                src="https://t4.ftcdn.net/jpg/07/98/90/05/360_F_798900572_Hw4UNsW7mKF2BTc9bEYmNaJFERMUZia0.jpg"
                alt="Profile"
                className="w-9 h-9 rounded-full border cursor-pointer bg-white object-cover"
                onClick={handleProfileClick}
                style={{ backgroundColor: 'white' }}
              />
              {showProfile && (
                <div className={
                  (dark ? 'bg-gray-800 text-white' : 'bg-white text-gray-900') +
                  ' absolute right-0 mt-2 w-80 rounded-md shadow-lg z-50 border border-gray-200 dark:border-gray-700 min-w-[320px] max-w-xs'
                }>
                  <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700 break-words">
                    <div className="font-semibold text-lg">{user.displayName || 'No Name'}</div>
                    <div className="text-sm text-gray-500 dark:text-gray-300 break-words">{user.email}</div>
                  </div>
                  <button
                    onClick={handleLogout}
                    className="w-full text-left px-6 py-3 hover:bg-gray-100 dark:hover:bg-gray-700 text-red-600 dark:text-red-400"
                  >
                    Logout
                  </button>
                </div>
              )}
            </div>
          )}

          <button
            onClick={toggleDark}
            className={dark ? 'ml-4 px-3 py-2 rounded-md bg-gray-800 text-yellow-300' : 'ml-4 px-3 py-2 rounded-md bg-gray-200 text-gray-800'}
            title="Toggle dark mode"
          >
            {dark ? '🌙' : '☀️'}
          </button>
        </div>
      </div>
    </nav>
  )
}
