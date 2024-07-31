// --------------------------
// Section: Import Libraries
// --------------------------
import * as dotenv from "dotenv";
dotenv.config();

import { ChatGoogleGenerativeAI } from "@langchain/google-genai";

import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableLambda } from "@langchain/core/runnables";

// --------------------------
// Section: Declare Constant
// --------------------------
const apiKey = process.env.GOOGLE_API_KEY || "";

// --------------------------
// Section: Declare LLM
// --------------------------
const model = new ChatGoogleGenerativeAI({
  apiKey: apiKey,
  model: "gemini-1.5-flash",
  temperature: 0,
});

// --------------------------
// Section: Create a string output parser
// --------------------------
const outputParser = new StringOutputParser();

// --------------------------
// Section: Create Base Prompt
// --------------------------
/* If you want to shorten the joke add "No yapping" but the joke will be stale. */
const prompt = ChatPromptTemplate.fromTemplate(`
    You are a comedian called Cody Mike.

    Your goal is to tell short but funny joke (1 joke per request) based on the provided topic by the user. No yapping.
  
    User topic: {input}.
    `);

// --------------------------
// Section: Create Base Chain
// --------------------------
const chain = prompt.pipe(model).pipe(outputParser);
const response1 = await chain.invoke({
  input: "bears",
});
console.log(response1);

// --------------------------
// Section: Create Evaluation Prompt (You can name the prompt to whatever but I will call it Evaluation Prompt)
// --------------------------
const evaluationPrompt = ChatPromptTemplate.fromTemplate(`
  You are a joke evaluator called Jake Eval.

  Your goals: 
  - Evaluate the provided joke whether is it funny or not.
  - Give explanation on the joke.
  - Suggest a joke that improved on the original joke.

  Joke: {joke}.
  `);

// --------------------------
// Section: Create Evaluation Chain using RunnableLamda
// --------------------------
/* We have setup BasePrompt, BaseChain, EvaluationPrompt, the final Chain - EvaluationChain is the most important.
The EvaluationChain will be the chain that runs everything. The process goes as followed:
1. We define a function that takes in an input, this input will be invoked by the BaseChain and give us our result
(remember to use await). We then return this result and feed it as an input to the joke property of the 2nd chain.
2. After getting the input for the 2nd chain which is the output of the 1st chain, we pipe it through the 2nd prompt.
We then continue as normal by pipe to the model and to the outputParser.
3. We call the EvaluationChain with the initial input of the 1st chain and this will kick start everything. */
const evaluationChain = new RunnableLambda({
  func: async (input) => {
    const result = await chain.invoke(input);
    return { joke: result };
  },
})
  .pipe(evaluationPrompt)
  .pipe(model)
  .pipe(outputParser);
const response2 = await evaluationChain.invoke({ input: "bears" });
console.log(response2);

// --------------------------
// Section: Synopsis
// --------------------------
/* 
  Learn:
  - Using RunnableLamda for chaining multiple chains
  Question:
*/
