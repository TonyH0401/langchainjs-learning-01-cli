// --------------------------
// Section: Import Libraries
// --------------------------
/* Library not from Langchain */
import * as dotenv from "dotenv";
dotenv.config();
import readline from "readline";
/* "@langchain/google-genai" Library */
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
/* "@langchain/core" Library */
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { MessagesPlaceholder } from "@langchain/core/prompts";
import { AIMessage, HumanMessage } from "@langchain/core/messages";
/* "@langchain/communit" Library */
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
/* Other Libraries */
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createOpenAIFunctionsAgent, AgentExecutor } from "langchain/agents";
import { createRetrieverTool } from "langchain/tools/retriever";

// --------------------------
// Section: Define LLM(s)
// --------------------------
const model = new ChatGoogleGenerativeAI({
  apiKey: process.env.GOOGLE_API_KEY || "",
  model: "gemini-1.5-flash",
  temperature: 0,
});

// --------------------------
// Section: Define (Agent) Prompt
// --------------------------
const prompt = ChatPromptTemplate.fromMessages([
  ("system",
  `
    You are a helpful assistant called Max.
    
    Your goal is to answer the user question and assist the user's need. No yapping.`),
  new MessagesPlaceholder("chat_history"),
  ("human", "{input}"),
  new MessagesPlaceholder("agent_scratchpad"),
]);

// --------------------------
// Section: Create and Assign Tools
// --------------------------
/* (Online) Search Tool */
const searchTool = new TavilySearchResults({
  apiKey: process.env.TAVILY_API_KEY,
  maxResults: 1,
});
/* Retriever Tool */
const loader = new CheerioWebBaseLoader(
  "https://docs.smith.langchain.com/user_guide"
);
const rawDocs = await loader.load();
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});
const docs = await splitter.splitDocuments(rawDocs);
const vectorstore = await MemoryVectorStore.fromDocuments(
  docs,
  new GoogleGenerativeAIEmbeddings()
);
const retriever = vectorstore.asRetriever();
const retrieverTool = createRetrieverTool(retriever, {
  name: "langsmith_search",
  description: `
  You are an expert at searching information in documents.

  Your goal is to search for information about LangSmith. 
  For any questions about LangSmith, you must use this tool! 
  No yapping.`,
});
/* Assigning Tools */
const tools = [searchTool, retrieverTool];

// --------------------------
// Section: Create Agent and AgentExecutor
// --------------------------
/* Create Agent */
const agent = await createOpenAIFunctionsAgent({
  llm: model,
  prompt: prompt,
  tools: tools,
});
/* Create Agent Executor */
const agentExecutor = new AgentExecutor({
  agent: agent,
  tools: tools,
});

// --------------------------
// Section: Initialize Chat History Storing Variable (not recommend)
// --------------------------
const chatHistory = [];

// --------------------------
// Section: Define Langfuse Tracer
// --------------------------
import { CallbackHandler } from "langfuse-langchain";
const langfuseHandler = new CallbackHandler({
  publicKey: process.env.LANGFUSE_PUB,
  secretKey: process.env.LANGFUSE_SEC,
  baseUrl: "https://cloud.langfuse.com",
});

// --------------------------
// Section: Init
// --------------------------
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});
const askQuestion = () => {
  rl.question("User: ", async (input) => {
    const response = await agentExecutor.invoke(
      {
        input: input,
        chat_history: chatHistory,
      },
      { callbacks: [langfuseHandler] }
    );

    console.log("Agent: ", response.output);
    chatHistory.push(new HumanMessage(input));
    chatHistory.push(new AIMessage(response.output));
    askQuestion();
  });
};
askQuestion();

// --------------------------
// Section: Synopsis
// --------------------------
/* Things it can do:
  - Answer based on the trained data of Google and OpenAI
  - Using a search tools to look it up.
  - Headup, most of Langchain function are more suitable with OpenAI, I tried cross/mix with different LLM(s) and functions, 
  they do not work, Langfuse tracing finds no traces of them being used anywhere.
*/
