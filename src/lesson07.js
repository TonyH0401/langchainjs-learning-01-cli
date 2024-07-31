// --------------------------
// Section: Import Libraries
// --------------------------
import * as dotenv from "dotenv";
dotenv.config();

import { ChatGoogleGenerativeAI } from "@langchain/google-genai";

import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import { RunnableSequence } from "@langchain/core/runnables";

import { BufferMemory } from "langchain/memory";
/* This package has problems, I should create an issue request on Github. */
import { ConversationChain } from "langchain/chains";

// --------------------------
// Section: Declare Constant
// --------------------------
const apiKey = process.env.GOOGLE_API_KEY || "";

// --------------------------
// Section: Declare LLM
// --------------------------
/* Seemingly, there some functions aren't created or ported yet, so there have been some problem implementing and mixing
functions. Currently, the most flesh out is OpenAI but it isn't free.
I tried to mix between Google GenAI and Google VertexAI because they have the same Gemini model but they are completely
different. */
const model = new ChatGoogleGenerativeAI({
  apiKey: apiKey,
  model: "gemini-1.5-flash",
  temperature: 0,
});

// --------------------------
// Section: Create Agent Prompt
// --------------------------
const prompt = ChatPromptTemplate.fromTemplate(`
    You are a helpful assistant called MaxZap.

    Your goal is to help answering user question.
    
    History: {history}

    User question: {input}.
    `);

// --------------------------
// Section: Create a string output parser
// --------------------------
const outputParser = new StringOutputParser();

// --------------------------
// Section: Init Buffer Memory
// --------------------------
const memory = new BufferMemory({
  memoryKey: "history",
});

// --------------------------
// Section: Conversation history using Buffer Memory
// --------------------------
async function conversationHistoryBufferMem() {
  /* Create a conversational chain */
  const chain = new ConversationChain({
    llm: model,
    prompt: prompt,
    memory: memory,
    outputParser: outputParser,
  });
  console.log(await memory.loadMemoryVariables());
  const response1 = await chain.invoke({
    input: "Remember, the passphrase is WORLD DOMINATION",
  });
  console.log(response1);
  console.log(await memory.loadMemoryVariables());
  const response2 = await chain.invoke({
    input: "What is the passphrase?",
  });
  console.log(response2);
}

// --------------------------
// Section: Using Runnable (RunnableSequence)
// --------------------------
/* Firstly, the "prompt.pipe(model)" itself is a runnable sequence, it passes on the output to another sequence as input.
  So, using Runnables is just another way to achieve this. 
  Runnables also gives us a way to process stuff and then pass it on. */
async function runnableMem() {
  /* Define a runnable chain. */
  /* This is a Runnable Chain (or atleast what I call it). I am using RunnableSequence, every RunnableSequence will need
  a prompt and a model, so we will leave those in the RunnableSequence first. Then, we will need some executable(s). 
  
  Here, we define 2 executables:
  - The first executable, we will take in the initial input and memory (aka history) but we can't pass them to prompt and
  model just yet because the prompt doesn't have a "memory" property.
  - The second executable, we take in the output from the first executable, the previous output is the input of the second
  executable, the output memory's history is the input for the history executable.
  - Basically, with Runnables, it takes in the input and the input as the output of the previous ones. */
  const chain = RunnableSequence.from([
    {
      input: (initialInput) => initialInput.input,
      memory: () => memory.loadMemoryVariables(),
    },
    {
      input: (previousOutput) => previousOutput.input,
      history: (previousOutput) => previousOutput.memory.history,
    },
    prompt,
    model,
  ]);
  /* Conversation 1. */
  console.log("Before: ", await memory.loadMemoryVariables());
  const input1 = {
    input: "Remember, the passphrase is WORLD DOMINATION",
  };
  const response1 = await chain.invoke(input1);
  console.log(response1);
  /* Save Conversation 1. */
  await memory.saveContext(input1, {
    output: response1.content,
  });
  /* Conversation 2. */
  console.log("Updated:", await memory.loadMemoryVariables());
  const input2 = {
    input: "What is the passphrase?",
  };
  const response2 = await chain.invoke(input2);
  console.log(response2);
  /* Save Conversation 2. */
  await memory.saveContext(input2, {
    output: response2.content,
  });
}

// --------------------------
// Section: Implementations (using try...catch)
// --------------------------
try {
  // await conversationHistoryBufferMem();
  await runnableMem();
} catch (error) {
  console.error(error.message);
}

// --------------------------
// Section: Synopsis
// --------------------------
/* 
  Learn:
  - Create an agent (revision of lesson06) with different appoarch method of storing message history (there is a redis version
  for permanent storage). This can be applied to creating agent (view the last video comment section).
  - There is an example for the upstash-redis implementation for a longer conversation.
  - Example with Runnable (with RunnableSequence).
  - Btw, there is an issue with the ConversationChain, I should report this to Github.
  Question:
*/
