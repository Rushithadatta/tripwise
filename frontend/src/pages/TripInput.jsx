import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'


export default function TripInput() {
// ...existing code...
// Add state for roundtrip cost
const [roundtripCost, setRoundtripCost] = useState(null);
const [form, setForm] = useState({
	source: '',
	destination: '',
	num_people: '',
	days: '',
	budget: '',
	transport: '',
	hotel: '',
	food: ''
})
function onFoodChange(e) {
	setForm(prev => ({ ...prev, food: e.target.value }))
}
const navigate = useNavigate()



function onChange(e) {
	const { name, value } = e.target
	setForm(prev => ({ ...prev, [name]: value }))
}

function onTransportChange(e) {
	setForm(prev => ({ ...prev, transport: e.target.value }))
}

function onHotelChange(e) {
	setForm(prev => ({ ...prev, hotel: e.target.value }))
}




function onSubmit(e) {
e.preventDefault()
// simple validation
if (!form.source || !form.destination || !form.num_people) return alert('Enter source, destination, and number of people')


// Fetch itinerary and roundtrip cost from backend
fetch('http://localhost:5000/api/itinerary', {
	method: 'POST',
	headers: { 'Content-Type': 'application/json' },
	body: JSON.stringify(form)
})
	.then(res => res.json())
	.then(data => {
		if (data.roundtrip_cost !== undefined) {
			setRoundtripCost(data.roundtrip_cost);
		}
		// Navigate to itinerary page with form data
		navigate('/itinerary', { state: { form, itinerary: data } });
	})
	.catch(err => {
		alert('Failed to fetch itinerary');
	});
}


return (
	<div className="max-w-4xl mx-auto p-6">
		{roundtripCost !== null && (
			<div className="mb-4 p-3 rounded-lg bg-yellow-100 text-yellow-800 font-semibold">
				Estimated Roundtrip Cost: ₹{roundtripCost}
			</div>
		)}
		<h2 className="text-2xl font-bold mb-4">Create a trip</h2>
		<form onSubmit={onSubmit} className="grid grid-cols-1 md:grid-cols-2 gap-4">
			<input name="source" value={form.source} onChange={onChange} placeholder="Source (city)" className="p-3 rounded-lg border" />
			<input name="destination" value={form.destination} onChange={onChange} placeholder="Destination (city)" className="p-3 rounded-lg border" />
			<input name="num_people" type="number" min={1} value={form.num_people} onChange={onChange} placeholder="Number of people" className="p-3 rounded-lg border" />
			<input name="days" type="number" min={1} value={form.days} onChange={onChange} placeholder="Days" className="p-3 rounded-lg border" />
			<input name="budget" value={form.budget} onChange={onChange} placeholder="Budget (INR)" className="p-3 rounded-lg border" />

			{/* Transport type dropdown */}
			<div>
				<label className="block mb-1 font-medium">Transport type</label>
				<select name="transport" value={form.transport} onChange={onTransportChange} className="p-3 rounded-lg border w-full">
					<option value="">Select transport</option>
					<option value="flight">Flight</option>
					<option value="train">Train</option>
					<option value="bus">Bus</option>
				</select>
			</div>

			{/* Hotel type dropdown */}
			<div>
				<label className="block mb-1 font-medium">Hotel type</label>
				<select name="hotel" value={form.hotel} onChange={onHotelChange} className="p-3 rounded-lg border w-full">
					<option value="">Select hotel type</option>
					<option value="ac">AC</option>
					<option value="non-ac">NON AC</option>
				</select>
			</div>

			{/* Food type dropdown */}
			<div>
				<label className="block mb-1 font-medium">Food type</label>
				<select name="food" value={form.food} onChange={onFoodChange} className="p-3 rounded-lg border w-full">
					<option value="">Select food type</option>
					<option value="veg">Veg</option>
					<option value="non-veg">Non Veg</option>
				</select>
			</div>

			<div className="md:col-span-2 flex items-center justify-end">
				<button type="submit" className="px-6 py-3 rounded-lg bg-blue-600 text-white">Generate Itinerary</button>
			</div>
		</form>

		<div className="mt-6 text-sm text-gray-500">
			Tip: Enter your requirements, transport, and hotel type for a better itinerary.
		</div>
	</div>
)
}