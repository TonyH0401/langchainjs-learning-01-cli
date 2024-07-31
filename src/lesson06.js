// --------------------------
// Section: Import Libraries
// --------------------------
import * as dotenv from "dotenv";
dotenv.config();
import readline from "readline";

import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";

import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import { MessagesPlaceholder } from "@langchain/core/prompts";

import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";

import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

import { MemoryVectorStore } from "langchain/vectorstores/memory";

import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";

import {
  AgentExecutor,
  createOpenAIFunctionsAgent,
  createToolCallingAgent,
} from "langchain/agents";

import { createRetrieverTool } from "langchain/tools/retriever";

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
// Section: Chat History (Template)
// --------------------------
const chatHistory = [];

// --------------------------
// Section: Create Agent Prompt
// --------------------------
const prompt = ChatPromptTemplate.fromMessages([
  ("system",
  `
    You are a helpful assistant called MaxZap.

    Your goal is to help answering user question.`),
  new MessagesPlaceholder("chat_history"),
  ("human", "{input}"),
  new MessagesPlaceholder("agent_scratchpad"),
]);

// --------------------------
// Section: Load data and Init Vector storage
// --------------------------
/* Load data */
const loader = new CheerioWebBaseLoader(
  "https://js.langchain.com/v0.1/docs/expression_language/"
);
const rawDocs = await loader.load();
/* Split data into smaller chunks */
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 200,
  chunkOverlap: 20,
});
const docs = await splitter.splitDocuments(rawDocs);
/* Create a vector storage */
const vectorstore = await MemoryVectorStore.fromDocuments(
  docs,
  new GoogleGenerativeAIEmbeddings()
);
/* Retrieve data */
const retriever = vectorstore.asRetriever({ k: 2 });

// --------------------------
// Section: Create and Assign Tool(s)
// --------------------------
const searchTool = new TavilySearchResults({
  apiKey: process.env.TAVILY_API_KEY,
  maxResults: 5,
});
const retrieverTool = createRetrieverTool(retriever, {
  name: "lcel_search",
  description: ` Use this tool when searching for information about Langchain Expression Language (LCEL).
    For any questions about Langchain Expression Language (LCEL), you must use this tool!`,
});
/* Because we defined tools as an array, you can assign multiple for an agent. */
const tools = [searchTool, retrieverTool];

// --------------------------
// Section: Create Agent
// --------------------------
/* Seemingly, there some functions aren't created or ported yet, so there have been some problem implementing and mixing
functions. Currently, the most flesh out is OpenAI but it isn't free.
I tried to mix between createOpenAIFunctionsAgent() and createToolCallingAgent() but because they are from different
models or some different errors, they aren't working at all (1 kinda work but doesn't call the tool, the other don't).
I should test it with different models on Python. */
const agent = await createOpenAIFunctionsAgent({
  llm: model,
  prompt: prompt,
  tools: tools,
});

// --------------------------
// Section: Execute Agent
// --------------------------
const agentExecutor = new AgentExecutor({
  agent: agent,
  tools: tools,
});

// --------------------------
// Section: Langfuse implementation
// --------------------------
import { CallbackHandler } from "langfuse-langchain";
const langfuseHandler = new CallbackHandler({
  publicKey: process.env.LANGFUSE_PUB,
  secretKey: process.env.LANGFUSE_SEC,
  baseUrl: "https://cloud.langfuse.com",
});

// --------------------------
// Section: Implementations (using try...catch)
// --------------------------
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});
const askQuestion = () => {
  rl.question("> User: ", async (input) => {
    const response = await agentExecutor.invoke(
      {
        input: input,
        chat_history: chatHistory,
      },
      { callbacks: [langfuseHandler] }
    );
    console.log(">> Agent: ", response.output);
    chatHistory.push(new HumanMessage(input));
    chatHistory.push(new AIMessage(response.output));
    askQuestion();
  });
};
askQuestion();

// --------------------------
// Section: Synopsis
// --------------------------
/* 
  Learn:
  - Create an Agent that can perform QnA. However, there is a slight problem, the tools can be read but it can't be recognized.
  There is something about the mixture of different functions and LLM(s) that disabling the functions to work well. Most
  functions in Langchain seems to work very well with OpenAI but OpenAI is not free. I will try another method using Python.
    - It should be able to look information up using tavily search (tavily search when tested actually work).
    - It should be able to retrieve data.
  - Implementing Langfuse to trace LLM process and that is how I know that the functions aren't being recognized.
  - Create conversational chat with history.
  Question:
*/
