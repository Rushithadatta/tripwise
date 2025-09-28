import React, { useEffect, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import TripCard from '../components/TripCard'
import { generateItinerary } from '../utils/api'

export default function Itinerary() {
  const [recommendations, setRecommendations] = useState({});
  const [activeRec, setActiveRec] = useState({ day: null, placeIdx: null });
  const [latestRecommendation, setLatestRecommendation] = useState(null);

  function handlePlaceClick(dayIdx, placeIdx, placeName) {
    // If already active, toggle off
    if (activeRec.day === dayIdx && activeRec.placeIdx === placeIdx) {
      setActiveRec({ day: null, placeIdx: null });
      setLatestRecommendation(null);
      return;
    }
    setActiveRec({ day: dayIdx, placeIdx });
    fetch(`http://localhost:5000/api/recommend?place=${encodeURIComponent(placeName)}`)
      .then(res => res.json())
      .then(data => {
        setRecommendations(prev => ({ ...prev, [`${dayIdx}-${placeIdx}`]: data }));
        setLatestRecommendation(data);
      })
      .catch(() => {
        const emptyData = { places: [], hotels: [], food: [] };
        setRecommendations(prev => ({ ...prev, [`${dayIdx}-${placeIdx}`]: emptyData }));
        setLatestRecommendation(emptyData);
      });
  }

  const { state } = useLocation();
  const navigate = useNavigate();
  const form = state?.form || { source: 'Hyderabad', destination: 'Jaipur', people: 2, days: 3, budget: 20000 };

  const [loading, setLoading] = useState(true);
  const [itinerary, setItinerary] = useState([]);
  const [costs, setCosts] = useState({ transport: 0, hotels: 0, activities: 0, total: 0 });
  const [roundTrip, setRoundTrip] = useState(0);
  const [expanded, setExpanded] = useState([]);
  const [rawJson, setRawJson] = useState(null);
  const [remainingBudget, setRemainingBudget] = useState(null);

  useEffect(() => {
    async function fetchItinerary() {
      setLoading(true);
      try {
        const res = await generateItinerary(form);
        setRawJson(res);
        setItinerary(res.days || []);
        setCosts(res.costs || { transport: 0, hotels: 0, activities: 0, total: form.budget });
        setRoundTrip(res.roundtrip_cost || 0);
        setExpanded(Array((res.days || []).length).fill(false));
        if (typeof res.remaining_budget === 'number') {
          setRemainingBudget(res.remaining_budget);
          if (res.remaining_budget < 0) {
            setTimeout(() => {
              alert('Your remaining budget is negative. Please extend your budget to continue.');
              navigate('/plan');
            }, 500);
          }
        }
      } catch (e) {
        setRawJson(null);
        setItinerary([]);
        setCosts({ transport: 0, hotels: 0, activities: 0, total: form.budget });
      }
      setLoading(false);
    }
    fetchItinerary();
  }, [form, navigate]);

  function toggleExpand(idx) {
    setExpanded(prev => prev.map((v, i) => (i === idx ? !v : v)));
  }

  const mapUrl = `https://www.google.com/maps?q=${encodeURIComponent(form.destination)}&output=embed`;

  return (
    <div className="max-w-6xl mx-auto p-6 grid grid-cols-1 lg:grid-cols-3 gap-6">
      <div className="lg:col-span-2 space-y-4">
        <div className="bg-white rounded-2xl p-6 shadow-md">
          <h3 className="text-xl font-semibold">
            Itinerary — {form.source} → {form.destination}
          </h3>
          <p className="text-sm text-gray-500 mt-1">
            {form.people} people • {form.days} days • Budget: ₹{form.budget}
          </p>
        </div>

        <div className="space-y-4">
          {loading ? (
            <div className="text-center py-8 text-blue-600 font-semibold">
              Loading itinerary...
            </div>
          ) : itinerary.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              No itinerary found.
            </div>
          ) : (
            <div>
              <div className="mb-4 text-lg font-semibold text-blue-700">
                Round-trip cost: ₹{roundTrip}
              </div>
              {itinerary.map((it, idx) => (
                <div key={idx} className="bg-white rounded-xl shadow-md mb-4">
                  <button
                    type="button"
                    className="w-full text-left px-6 py-4 flex justify-between items-center focus:outline-none"
                    onClick={() => toggleExpand(idx)}
                  >
                    <span className="font-semibold text-lg">
                      Day {it.day ?? idx + 1}
                    </span>
                    <span className="text-blue-600">
                      {expanded[idx] ? "▲" : "▼"}
                    </span>
                  </button>

                  {expanded[idx] && (
                    <div className="px-6 pb-4 pt-2 space-y-4">
                      <ul className="space-y-3">
                        {(it.attractions || []).map((place, i) => (
                          <li key={i} className="border rounded-lg p-4 bg-gray-50 shadow-sm">
                            <div className="flex flex-col gap-3">
                              {/* Place name clickable → Google Maps */}
                              <span
                                className="font-semibold text-lg text-blue-700 hover:underline cursor-pointer"
                                onClick={() =>
                                  window.open(
                                    `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(place.name)}`,
                                    "_blank"
                                  )
                                }
                              >
                                {place.name}
                              </span>

                              {/* Place details in compact grid */}
                              <div className="grid grid-cols-2 md:grid-cols-3 gap-x-6 gap-y-2 text-sm text-gray-700">
                                <div>
                                  <span className="font-medium block">Timings:</span>
                                  {Array.isArray(place.timings) ? (
                                    <ul className="ml-4 list-disc">
                                      {place.timings.map((t, idx) => (
                                        <li key={idx} className="text-gray-700">{t}</li>
                                      ))}
                                    </ul>
                                  ) : typeof place.timings === "string" &&
                                    (place.timings.includes(",") || place.timings.includes(";")) ? (
                                    <ul className="ml-4 list-disc">
                                      {place.timings.split(/[,;]/).map((t, idx) => (
                                        <li key={idx} className="text-gray-700">{t.trim()}</li>
                                      ))}
                                    </ul>
                                  ) : (
                                    <span className="ml-2 text-gray-700">{place.timings}</span>
                                  )}
                                </div>

                                <div>
                                  <span className="font-medium">Best Time:</span> {place.best_time}
                                </div>
                                <div>
                                  <span className="font-medium">Duration:</span> {place.time_spent}
                                </div>
                                <div>
                                  <span className="font-medium">Cost:</span> ₹{place.cost}
                                </div>
                              </div>

                              {/* Nearby Places button */}
                              <div>
                                <button
                                  className="mt-2 px-3 py-1 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700 transition"
                                  onClick={() => handlePlaceClick(idx, i, place.name)}
                                >
                                  Show Nearby Places
                                </button>
                              </div>

                              {/* Recommendation popup */}
                              {activeRec.day === idx && activeRec.placeIdx === i && recommendations[`${idx}-${i}`] && (
                                <div
                                  className="fixed z-50 top-1/4 left-1/2 md:left-[58%] bg-white border border-blue-400 rounded-xl shadow-2xl p-6 min-w-[260px] max-w-xs md:max-w-sm animate-fade-in overflow-y-auto max-h-[70vh]"
                                  style={{ transform: "translateX(10%)" }}
                                >
                                  <div className="flex justify-between items-center mb-2">
                                    <h5 className="font-semibold text-blue-700">Near by Recommendations</h5>
                                    <button
                                      className="text-gray-500 hover:text-red-500 text-lg font-bold"
                                      onClick={() => setActiveRec({ day: null, placeIdx: null })}
                                    >
                                      &times;
                                    </button>
                                  </div>
                                  <div className="space-y-2">
                                    {recommendations[`${idx}-${i}`].places?.length > 0 && (
                                      <div>
                                        <span className="font-medium">Places:</span>
                                        <ul className="list-disc ml-6 text-gray-700">
                                          {recommendations[`${idx}-${i}`].places.map((rec, j) => (
                                            <li key={j}>{rec}</li>
                                          ))}
                                        </ul>
                                      </div>
                                    )}
                                    {recommendations[`${idx}-${i}`].hotels?.length > 0 && (
                                      <div>
                                        <span className="font-medium">Hotels:</span>
                                        <ul className="list-disc ml-6 text-gray-700">
                                          {recommendations[`${idx}-${i}`].hotels.map((rec, j) => (
                                            <li key={j}>{rec}</li>
                                          ))}
                                        </ul>
                                      </div>
                                    )}
                                    {recommendations[`${idx}-${i}`].food?.length > 0 && (
                                      <div>
                                        <span className="font-medium">Food:</span>
                                        <ul className="list-disc ml-6 text-gray-700">
                                          {recommendations[`${idx}-${i}`].food.map((rec, j) => (
                                            <li key={j}>{rec}</li>
                                          ))}
                                        </ul>
                                      </div>
                                    )}
                                  </div>
                                </div>
                              )}
                            </div>
                          </li>
                        ))}
                      </ul>

                      {/* Extra day-level info */}
                      {it.hotel && (
                        <div className="text-sm text-green-700">
                          <span className="font-semibold">Hotel:</span> {it.hotel.name} (₹{it.hotel.cost})
                        </div>
                      )}
                      {it.restaurant && (
                        <div className="text-sm text-purple-700">
                          <span className="font-semibold">Restaurant:</span> {it.restaurant.name} (₹{it.restaurant.cost})
                        </div>
                      )}
                      {it.local_travel && (
                        <div className="text-sm text-blue-700">
                          <span className="font-semibold">Local Travel:</span> {it.local_travel.services}
                        </div>
                      )}
                      {it.emergency_services && (
                        <div className="text-sm text-red-700">
                          <span className="font-semibold">Emergency Services:</span>
                          <div className="overflow-x-auto mt-1">
                            <table className="min-w-[300px] w-full border border-red-300 bg-white text-gray-800 rounded-md">
                              <thead>
                                <tr className="bg-red-50">
                                  <th className="px-2 py-1 text-left font-semibold">Service</th>
                                  <th className="px-2 py-1 text-left font-semibold">Telephone</th>
                                  <th className="px-2 py-1 text-left font-semibold">Timings(24/7)</th>
                                </tr>
                              </thead>
                              <tbody>
                                {Object.entries(it.emergency_services).map(([key, val], j) => (
                                  <tr key={j} className="border-t border-red-100">
                                    <td className="px-2 py-1">{val.name || key.replace('_', ' ')}</td>
                                    <td className="px-2 py-1">{val.phone}</td>
                                    <td className="px-2 py-1">{val["24_7"] ? val["24_7"] : val.timings || "-"}</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      <aside className="space-y-4">
        <div className="bg-white rounded-2xl p-4 shadow-md">
          <h4 className="font-semibold">Cost breakdown</h4>
          <div className="mt-2 text-sm text-gray-700 font-semibold">Total Cost: ₹{costs.total}</div>
          <div className="mt-2 text-sm text-green-700 font-semibold">Remaining Budget: ₹{remainingBudget !== null ? remainingBudget : '...'}</div>
          <div className="mt-2 text-sm text-blue-700 font-semibold">Round Trip: ₹{roundTrip}</div>
        </div>

        <div className="bg-white rounded-2xl p-4 shadow-md">
          <h4 className="font-semibold">Map</h4>
          <div className="mt-3 rounded-lg overflow-hidden">
            <iframe
              title="Google Maps"
              src={mapUrl}
              width="100%"
              height="200"
              style={{ border: 0 }}
              allowFullScreen
              loading="lazy"
              referrerPolicy="no-referrer-when-downgrade"
            />
          </div>
        </div>
      </aside>
    </div>
  );
}
