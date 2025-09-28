import React from 'react'


export default function FeatureCard({ icon, title, desc }) {
return (
<div className="bg-white rounded-2xl p-6 shadow-md hover:shadow-xl transition">
<div className="text-3xl mb-4">{icon}</div>
<h3 className="text-xl font-semibold mb-2">{title}</h3>
<p className="text-gray-500">{desc}</p>
</div>
)
}