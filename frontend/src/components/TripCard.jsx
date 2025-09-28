import React from 'react'


export default function TripCard({ day, activities }) {
return (
<div className="bg-white rounded-2xl p-6 shadow-md">
<h4 className="font-semibold mb-3">{day}</h4>
<ul className="space-y-2 text-sm text-gray-600">
{activities.map((a, i) => (
<li key={i}>• {a}</li>
))}
</ul>
</div>
)
}