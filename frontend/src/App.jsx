import React from 'react'
import { Routes, Route } from 'react-router-dom'
import LandingPage from './pages/LandingPage'
import Login from './pages/Login'
import TripInput from './pages/TripInput'
import Itinerary from './pages/Itinerary'
import Navbar from './components/Navbar'


export default function App() {
return (
<div className="min-h-screen flex flex-col">
<Navbar />
<main className="flex-1">
<Routes>
<Route path="/" element={<LandingPage />} />
<Route path="/login" element={<Login />} />
<Route path="/plan" element={<TripInput />} />
<Route path="/itinerary" element={<Itinerary />} />
</Routes>
</main>
</div>
)
}