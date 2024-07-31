// --------------------------
// Section: Import Libraries
// --------------------------
import * as dotenv from "dotenv";
dotenv.config();
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
/* This library has a problem with "punycode", I should create an issue on Github. */
import { ConversationChain } from "langchain/chains";

// --------------------------
// Section: Declare LLM
// --------------------------
const model = new ChatGoogleGenerativeAI({
  apiKey: process.env.GOOGLE_API_KEY,
  model: "gemini-1.5-flash",
  temperature: 0,
});

// --------------------------
// Section: Create Agent Prompt
// --------------------------
const prompt = ChatPromptTemplate.fromTemplate(`
    You are a helpful assistant called MaxZap.

    Your goal is to help answering user question.

    User question: {input}.
    `);

// --------------------------
// Section: Create a string output parser
// --------------------------
const outputParser = new StringOutputParser();

// --------------------------
// Section: Testing
// --------------------------
const chain = prompt.pipe(model).pipe(outputParser);
const res = await chain.invoke({
  input: "What is your name? What is the capital of South Africa?",
});
console.log(res);

// --------------------------
// Section: Synopsis
// --------------------------
/* I was testing a "warning" with "ConversationChain" where there was a deprecate warning for "ConversationChain"
about the "punycode". I have located the error it isn't because of Node but with the package + Node version (Nodev20 is fine). 
I should raise an issue on Github. */
