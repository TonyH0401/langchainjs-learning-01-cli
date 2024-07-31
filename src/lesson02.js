// --------------------------
// Section: Import Libraries
// --------------------------
import * as dotenv from "dotenv";
dotenv.config();
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatPromptTemplate } from "@langchain/core/prompts";

// --------------------------
// Section: Declare Constant
// --------------------------
const apiKey = process.env.GOOGLE_API_KEY || "";

// --------------------------
// Section: Implementations (using try...catch)
// --------------------------
try {
  // --------------------------
  // Section: Verify LLM API key
  // --------------------------
  if (!apiKey) throw new Error("> LLM apiKey not found!");

  // --------------------------
  // Section: Declare LLM
  // --------------------------
  const model = new ChatGoogleGenerativeAI({
    apiKey: apiKey,
    model: "gemini-1.5-flash",
    temperature: 0,
  });

  // --------------------------
  // Section: Create Prompt Template
  // --------------------------
  /* Create a prompt (in bulk format) using fromTemplate() */
  // const prompt = ChatPromptTemplate.fromTemplate(
  //   `You are a comedian.
  //    Tell a joke based on the following word {input}.`
  // );
  /* Display the prompt in complete format */
  // console.log(await prompt.format({ input: "chicken" }));
  /* Create a prompt (in separate format) using fromMessages() */
  const prompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      `You are an excellent comedian. Your jokes are short but meaningful.
      Tell a joke based on the following word provided by the user.`,
    ],
    ["human", "{input}"],
  ]);

  // --------------------------
  // Section: Create Chain
  // --------------------------
  /* You can chain multiple chains together (chaining) 
  or 1 chain can be used in the creation of another chain (nested).
  */
  const chain = prompt.pipe(model);

  // --------------------------
  // Section: Invoke Chain
  // --------------------------
  const response = await chain.invoke({
    input: "dog",
  });
  console.log(response);
} catch (error) {
  console.error(error.message);
}

// --------------------------
// Section: Synopsis
// --------------------------
/* 
  Learn:
  - How to create a prompt.
  - How many type of different prompting method.

  Question:
  - Format the response.
*/
