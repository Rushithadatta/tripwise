Prototype Demo :

🛫 TripWise : Smart AI Trip Planner

🔹 Overview:
TripWise is a smart AI-powered trip planning website that helps travelers generate personalized, budget-friendly, and day-wise itineraries.
Unlike existing apps, TripWise focuses on budget-first planning, smart nearby suggestions, and integration of local services (guides, rentals, emergency contacts).

🔹 Problem Statement:
Travelers struggle to manually plan trips across scattered platforms (transport, hotels, food, attractions).
Existing apps give static recommendations without considering budget, food preferences, or must-visit places.
Local services like rentals and emergency contacts are missing.
👉 This leads to wasted time, overspending, and missed cultural experiences.

🔹 Features:
✅ Takes inputs like source, destination, people, days, budget, transport, hotel type (AC/Non-AC), and food (Veg/Non-Veg).
✅ Generates AI-optimized day-wise itineraries.
✅ Provides transport, hotels, food, attractions, and hidden gems.
✅ Includes local services like bike rentals, camera rentals, emergency contacts (hospital, police station).
✅ Real-time trip tracking + nearby recommendations.
✅ Export plans to PDF/Excel for offline use.

🔹 User Journey:
Login/Signup – via Google, email, or guest mode.
Enter Trip Inputs – source, destination, budget, number of people, days.
Set Preferences – transport, hotel, food, must-visit places.
AI Itinerary Generation – budget-based, day-wise plan with nearby suggestions.
Customize & Share – swap attractions, adjust budget/time, export itinerary.
Local Support – guides, rentals, hospitals, police stations.

🔹 Tech Stack:
Frontend: React.js, TailwindCSS / Material UI, Redux/Context API, React Router
Backend: Node.js + Express, Python (Flask/FastAPI) for AI/ML
Database: MongoDB / Firebase
APIs: Google Places, Skyscanner/Amadeus, IRCTC, OYO/Booking.com, Zomato/Yelp
AI/ML: Recommendation engine, budget optimizer, NLP chatbot
Maps: Google Maps API, GPS tracking

🔹 Roadmap:
Phase 1 (MVP): Login + Source/Destination + Basic itinerary
Phase 2: Personalization (transport, hotel, food preferences)
Phase 3: Local support & emergency integration
Phase 4: Smart features (AI chatbot, offline export, gamification)

🔹 Example Flow
Input: Hyderabad → Jaipur, 3 people, 5 days, ₹20,000 budget
Preferences: Train, Budget AC hotel, Veg food
Must-visit: Amber Fort, Hawa Mahal

Generated Plan:
Day 1: Hawa Mahal + City Palace + Jantar Mantar
Day 2: Amber Fort + Jaigarh Fort + Jal Mahal
Day 3: Food walk + bazaars
Day 4: Pushkar day trip
Day 5: Shopping + return
Includes cost breakdown + emergency contacts + export option.

🔹 Unique Selling Points:
🏷️ Budget-first planning
📍 Nearby smart suggestions
🚑 Local services & emergency integration
🎭 Cultural & hidden experiences
📄 Offline export

🔹 Installation & Setup:

# Clone the repository
git clone https://github.com/your-username/tripwise.git
cd tripwise

# Install frontend dependencies
cd frontend
npm install
npm start

# Install backend dependencies
cd ../backend
npm install
npm run dev

🔹 Contributing:
We welcome contributions!
Fork the repo
Create a feature branch (git checkout -b feature-name)
Commit changes (git commit -m "Added feature")
Push and create a PR
