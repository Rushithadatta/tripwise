import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import axios from "axios";
import { spawn } from "child_process";
import path from "path";
import { fileURLToPath } from "url";

// Load environment variables
dotenv.config();

// Express setup
const app = express();
const PORT = process.env.PORT || 5000;

app.use(cors());
app.use(express.json());

// __dirname workaround (ESM)
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// === Recommendation Route ===
app.get("/api/recommend", async (req, res) => {
  const place = req.query.place;
    console.log(`[RECOMMEND] Request received for place: ${place}`);
  if (!place) {
    return res.status(400).json({ error: "Missing 'place' query parameter" });
  }

  try {
      console.log(`[RECOMMEND] Spawning Python process for: ${place}`);
    // Resolve Python script path correctly
    const scriptPath = path.resolve(__dirname, "../model_training/main.py");

    // Arguments to send into Python
    const pyArgs = JSON.stringify({
      recommend: true,
      place: place,
    });

    const pythonProcess = spawn("python", [scriptPath], {
      stdio: ["pipe", "pipe", "pipe"],
      cwd: path.resolve(__dirname, "../model_training"),
    });

    // Send JSON to Python via stdin
    pythonProcess.stdin.write(pyArgs);
    pythonProcess.stdin.end();

    let result = "";

    pythonProcess.stdout.on("data", (data) => {
      result += data.toString();
    });

    pythonProcess.stderr.on("data", (data) => {
      console.error("Python error (recommend):", data.toString());
    });

    pythonProcess.on("close", () => {
        console.log(`[RECOMMEND] Python process exited`);
      try {
  const output = JSON.parse(result.trim());
  console.log("Recommended places for", place, ":", output);
  res.json(output);
      } catch (err) {
        console.error("JSON parse error (recommend):", err.message);
        res
          .status(500)
          .json({ error: "Failed to parse recommendations", raw: result });
      }
    });
  } catch (err) {
    console.error("Recommend model call error:", err.message);
    res.status(500).json({ error: "Failed to get recommendations" });
  }
});

// === Itinerary Route ===
app.post("/api/itinerary", async (req, res) => {
  const { source, destination, days, budget, num_people, transport, hotel, food } =
    req.body;
  const peopleCount = parseInt(num_people) || 1;

  let roundtrip_cost = null;
  // === Phase 1: Gemini API === 
    try { console.log("=== [Gemini Phase Start] ==="); 
   const prompt = `You are a travel cost calculator. Task: Estimate the round-trip travel cost 
   from ${source || ""} to ${destination || ""} in INR. Transport mode: ${transport || "any"}(Non-AC). 
    Number of people: ${peopleCount}. Rules: - Only return the final total cost as a plain integer 
   (no words, no explanation, no currency symbol). - Example output: 12500`; 
    const modelName = "gemini-1.5-flash-latest"; 
    console.log("[Gemini] Prompt:\n", prompt); 
   const geminiRes = await axios.post(`https://generativelanguage.googleapis.com/v1/models/${modelName}:generateContent?key=${process.env.GEMINI_API_KEY}`, 
   { contents:
   [ { role: "user", 
  parts: [{ text: prompt }], }, ], } ); 
  console.log("[Gemini] API raw response:", JSON.stringify(geminiRes.data, null, 2)); 
  const costText = geminiRes.data?.candidates?.[0]?.content?.parts?.[0]?.text?.trim() || ""; 
  console.log("[Gemini] Extracted cost text:", costText); 
   const costNum = parseInt(costText.replace(/[^\d]/g, "")); 
  console.log("[Gemini] Parsed cost number:", costNum); 
   if (!isNaN(costNum)) { 
   roundtrip_cost = costNum; 
   } else { 
   console.log("[Gemini → Fallback] Using default roundtrip cost..."); 
   roundtrip_cost = 1200 * peopleCount; 
   } console.log("=== [Gemini Phase End] ===\n"); 
   } catch (err) { 
   console.error("Gemini API error:", err.response?.status, err.response?.data || err.message); 
    roundtrip_cost = 1250 * peopleCount; }

  // === Phase 2: Python Model Arguments ===
  try {
    const pyArgs = {
      source: source || "",
      destination: destination || "",
      num_people: peopleCount,
      num_days: parseInt(days) || 1,
      budget: Math.max(parseInt(budget) - roundtrip_cost, 0),
      transport: transport || "",
      stay_type: hotel || "",
      food: food || "",
    };

    const scriptPath = path.resolve(
      __dirname,
      "../main_model_training/csp_tour_planner.py"
    );

    const pythonProcess = spawn("python", [scriptPath, JSON.stringify(pyArgs)], {
      stdio: ["pipe", "pipe", "pipe"],
    });

    let result = "";

    pythonProcess.stdout.on("data", (data) => {
      result += data.toString();
    });

    pythonProcess.stderr.on("data", (data) => {
      console.error("Python error (itinerary):", data.toString());
    });

    pythonProcess.on("close", () => {
      try {
        const itinerary = JSON.parse(result.trim());
        itinerary.roundtrip_cost = roundtrip_cost;
        res.json(itinerary);
      } catch (err) {
        console.error("JSON parse error (itinerary):", err.message);
        res
          .status(500)
          .json({ error: "Failed to parse itinerary from model", raw: result });
      }
    });
  } catch (err) {
    console.error("Model call error:", err.message);
    res.status(500).json({ error: "Failed to generate itinerary" });
  }
});

// === Health Check Route ===
app.get("/", (req, res) => {
  res.json({ message: "TripWise Backend Running 🚀" });
});

// Start server
app.listen(PORT, () => {
  console.log(`✅ Backend running at http://localhost:${PORT}`);
});
