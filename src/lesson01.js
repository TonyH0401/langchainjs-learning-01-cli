// --------------------------
// Section: Import Libraries
// --------------------------
import * as dotenv from "dotenv";
dotenv.config();
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";

// --------------------------
// Section: Declare Constant
// --------------------------
const googleApiKey = process.env.GOOGLE_API_KEY || "";

// --------------------------
// Section: Implementations (using try...catch)
// --------------------------
try {
  // --------------------------
  // Section: Verify LLM API key
  // --------------------------
  if (!googleApiKey) throw new Error("> LLM apiKey not found!");
  // --------------------------
  // Section: Declare LLM
  // --------------------------
  const model = new ChatGoogleGenerativeAI({
    apiKey: googleApiKey,
    model: "gemini-1.5-flash",
    temperature: 0,
    // verbose: true,
  });
  // --------------------------
  // Section: Reponse returned from the LLM
  // --------------------------
  const response = await model.invoke("Write a poem about AI");
  console.log(response);
  // const response = await model.batch(["hello", "How are you?"]);
  // console.log(response);
} catch (error) {
  console.error(error.message);
}

// --------------------------
// Section: Synopsis
// --------------------------
/* 
  Learn:
  - Initialize an LLM from API key.
  - Sending questions to LLM.

  Question:
  - Format the response for a better display.
*/
