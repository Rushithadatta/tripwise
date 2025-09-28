import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { 
  getAuth, 
  signInWithPopup, 
  GoogleAuthProvider, 
  createUserWithEmailAndPassword, 
  signInWithEmailAndPassword 
} from 'firebase/auth'
import { initializeApp } from 'firebase/app'
import { getAnalytics, isSupported } from "firebase/analytics";

// ✅ Correct Firebase Config
const firebaseConfig = {
  apiKey: "YOUR_FIREBASE_API_KEY",
  authDomain: "YOUR_DOMAIN",
  projectId: "smarttripai-f9c1e",
  storageBucket: "YOUR_STORAGE",
  messagingSenderId: "",
  appId: "",
  measurementId: ""
};

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);

// ✅ Load Analytics only in browser
isSupported().then((yes) => {
  if (yes) getAnalytics(app);
});


function Login() {
  const [isSignup, setIsSignup] = useState(false)
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [rePassword, setRePassword] = useState('')
  const [error, setError] = useState('')
  const navigate = useNavigate();

  async function handleGoogleLogin() {
    setError('')
    try {
      const provider = new GoogleAuthProvider()
  await signInWithPopup(auth, provider)
  navigate('/')
    } catch (e) {
      setError(e.message)
    }
  }

  async function handleEmailSubmit(e) {
    e.preventDefault()
    setError('')
    try {
      if (isSignup) {
        if (!name) {
          setError('Please enter your name')
          return
        }
        if (password !== rePassword) {
          setError('Passwords do not match')
          return
        }
        await createUserWithEmailAndPassword(auth, email, password)
        // Optionally update profile with name
        if (auth.currentUser) {
          await auth.currentUser.updateProfile({ displayName: name })
        }
      } else {
        await signInWithEmailAndPassword(auth, email, password)
      }
      navigate('/')
    } catch (e) {
      setError(e.message)
    }
  }

  return (
    <div className="max-w-md mx-auto mt-12 bg-white shadow-md rounded-2xl p-8">
      <h2 className="text-2xl font-bold mb-4">{isSignup ? 'Create an account' : 'Welcome back'}</h2>
      <form onSubmit={handleEmailSubmit}>
        {isSignup && (
          <input
            className="w-full p-3 border rounded-lg mb-3"
            placeholder="Name"
            value={name}
            onChange={e => setName(e.target.value)}
            type="text"
            required
          />
        )}
        <input
          className="w-full p-3 border rounded-lg mb-3"
          placeholder="Email"
          value={email}
          onChange={e => setEmail(e.target.value)}
          type="email"
          required
        />
        <input
          className="w-full p-3 border rounded-lg mb-3"
          placeholder="Password"
          type="password"
          value={password}
          onChange={e => setPassword(e.target.value)}
          required
        />
        {isSignup && (
          <input
            className="w-full p-3 border rounded-lg mb-4"
            placeholder="Re-enter Password"
            type="password"
            value={rePassword}
            onChange={e => setRePassword(e.target.value)}
            required
          />
        )}
        <button type="submit" className="w-full py-3 rounded-lg bg-blue-600 text-white mb-3">
          {isSignup ? 'Sign Up' : 'Login'}
        </button>
      </form>
      <button
        className="w-full py-3 rounded-lg flex items-center justify-center gap-2 border border-gray-300 hover:bg-gray-50"
        onClick={handleGoogleLogin}
        type="button"
      >
        <svg width="20" height="20" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
          <g clipPath="url(#clip0_17_40)">
            <path d="M47.5 24.5C47.5 22.6 47.3 20.8 47 19H24V29H37.1C36.5 32.1 34.4 34.6 31.5 36.2V42H39C44 38.3 47.5 32 47.5 24.5Z" fill="#4285F4"/>
            <path d="M24 48C30.6 48 36.2 45.7 39.9 42L31.5 36.2C29.6 37.3 27.2 38 24 38C18.7 38 14.1 34.4 12.5 29.7H4.7V35.7C8.4 42.1 15.6 48 24 48Z" fill="#34A853"/>
            <path d="M12.5 29.7C12.1 28.6 12 27.3 12 26C12 24.7 12.1 23.4 12.5 22.3V16.3H4.7C3.2 19.1 2.5 22.4 2.5 26C2.5 29.6 3.2 32.9 4.7 35.7L12.5 29.7Z" fill="#FBBC05"/>
            <path d="M24 14C27.2 14 29.6 15.1 31.1 16.5L39.9 8.1C36.2 4.6 30.6 2 24 2C15.6 2 8.4 7.9 4.7 14.3L12.5 20.3C14.1 15.6 18.7 14 24 14Z" fill="#EA4335"/>
          </g>
          <defs>
            <clipPath id="clip0_17_40">
              <rect width="48" height="48" fill="white"/>
            </clipPath>
          </defs>
        </svg>
        Continue with Google
      </button>
      {error && <div className="mt-3 text-red-500 text-sm">{error}</div>}
      <div className="mt-4 text-center">
        <button
          type="button"
          className="text-blue-600 underline text-sm"
          onClick={() => setIsSignup(s => !s)}
        >
          {isSignup ? 'Already have an account? Login' : "Don't have an account? Sign up"}
        </button>
      </div>
    </div>
  )
}

export default Login
 
