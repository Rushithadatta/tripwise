import React, { useEffect, useState } from 'react'
import FeatureCard from '../components/FeatureCard'
import { getAuth, onAuthStateChanged } from 'firebase/auth'


export default function LandingPage() {
	const [user, setUser] = useState(null)
	useEffect(() => {
		const auth = getAuth();
		const unsubscribe = onAuthStateChanged(auth, (u) => setUser(u));
		return () => unsubscribe();
	}, [])

	return (
		<div className="min-h-[70vh] flex flex-col items-center justify-center px-6 py-12 text-center">
			{/* Profile icon removed, only spacing remains for layout consistency */}
			<div className="w-full flex justify-end mb-2"></div>
			<h1 className="text-5xl font-bold text-blue-800 mb-4">Trip Wise🌍</h1>
			<p className="text-lg text-gray-600 mb-8 max-w-2xl">
				Smart AI Trip Planner — Personalized itineraries, budget-first planning, and local experiences.
			</p>
			<div className="flex gap-4">
				<a href="/plan" className="bg-blue-600 text-white px-6 py-3 rounded-xl text-lg shadow-lg hover:bg-blue-700 transition">Start Planning</a>
				{!user && (
					<a href="/login" className="border border-blue-600 text-blue-600 px-6 py-3 rounded-xl text-lg hover:bg-blue-50 transition">Login</a>
				)}
			</div>
			<div className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-8 max-w-5xl w-full">
				<FeatureCard icon={'✈️'} title="Smart Transport" desc="Flights, trains and buses optimized for cost and comfort." />
				<FeatureCard icon={'📍'} title="Hidden Gems" desc="Discover nearby attractions & cultural events." />
				<FeatureCard icon={'🏷️'} title="Budget Planner" desc="Stay inside budget with optimized day-wise plans." />
			</div>
		</div>
	)
}