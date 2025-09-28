export async function generateItinerary(formData) {
	try {
		const res = await fetch('http://localhost:5000/api/itinerary', {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
			},
			body: JSON.stringify(formData),
		})
		if (!res.ok) {
			throw new Error('Failed to fetch itinerary')
		}
		return await res.json()
	} catch (e) {
		alert('Could not generate itinerary. Please try again later.')
		return { itinerary: [] }
	}
}