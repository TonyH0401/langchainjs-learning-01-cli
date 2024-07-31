// --------------------------
// Section: Import Libraries
// --------------------------
import * as dotenv from "dotenv";
dotenv.config();

import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";

import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import { MessagesPlaceholder } from "@langchain/core/prompts";

import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";

import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

import { MemoryVectorStore } from "langchain/vectorstores/memory";

import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";

// --------------------------
// Section: Declare Constant
// --------------------------
const apiKey = process.env.GOOGLE_API_KEY || "";

// --------------------------
// Section: Create a string output parser
// --------------------------
const outputParser = new StringOutputParser();

// --------------------------
// Section: Create a vector storage (with a knowledge base)
// --------------------------
const createVectorStore = async () => {
  /* Loading the data (from a webpage). */
  const loader = new CheerioWebBaseLoader(
    "https://js.langchain.com/v0.1/docs/expression_language/"
  );
  const docs = await loader.load();
  /* Split the data into smaller chunks. */
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 200, // number of characters per chunk
    chunkOverlap: 20,
  });
  const splitDocs = await splitter.splitDocuments(docs);
  /* Create embedding for vector storage + Create in-memory vector storage. */
  const embeddings = new GoogleGenerativeAIEmbeddings();
  const vectorStore = await MemoryVectorStore.fromDocuments(
    splitDocs,
    embeddings
  );
  return vectorStore;
};

// --------------------------
// Section: Create retrieval chain (no history)
// --------------------------
const createChain = async (vectorStore) => {
  const model = new ChatGoogleGenerativeAI({
    apiKey: apiKey,
    model: "gemini-1.5-flash",
    temperature: 0,
  });
  const prompt = ChatPromptTemplate.fromTemplate(`
    You are an encyclopedia, you knows all the answers.

    Your goal is to answer the user's question.
    Do not make up information. No yapping.

    Context: {context}.
    Question: {input}.
    `);
  const chain = await createStuffDocumentsChain({
    llm: model,
    prompt: prompt,
  });
  const retriever = vectorStore.asRetriever({
    k: 2,
  });
  const retrievalChain = await createRetrievalChain({
    combineDocsChain: chain,
    retriever: retriever,
  });
  return retrievalChain;
};

// --------------------------
// Section: Test Chat History
// --------------------------
/* Create a test chat history in array format using Langchain history object. */
const chatHistory = [
  new HumanMessage("Hello"),
  new AIMessage("Hi, how can I help you?"),
  new HumanMessage(
    "My name is Aaron. I have a male 3 year-old German Shepherd dog name Donny."
  ),
  new AIMessage("Hi Aaron, how can I help you?"),
  new HumanMessage("What is LCEL?"),
  new AIMessage("LCEL stands for LangChain Expression Language"),
];

// --------------------------
// Section: Create retrieval chain (with history)
// --------------------------
const createChainHistory = async (vectorStore) => {
  const model = new ChatGoogleGenerativeAI({
    apiKey: apiKey,
    model: "gemini-1.5-flash",
    temperature: 0,
  });
  /* Can't use with fromTemplate(), you have to use it with fromMessages() */
  const prompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      "Answer the user's question based on the following context: {context}.",
    ],
    new MessagesPlaceholder("chat_history"),
    ["user", "{input}"],
  ]);
  const chain = await createStuffDocumentsChain({
    llm: model,
    prompt: prompt,
  });
  const retriever = vectorStore.asRetriever({
    k: 2,
  });
  const retrievalChain = await createRetrievalChain({
    combineDocsChain: chain,
    retriever: retriever,
  });
  return retrievalChain;
};

// --------------------------
// Section: Create retrieval chain improved (with history)
// --------------------------
/* Previously, we used a mixture of data from the vector storage + chat history, 
we will now improve it with data from vector storage + history + user question. */
const createChainHistoryImprove = async (vectorStore) => {
  const model = new ChatGoogleGenerativeAI({
    apiKey: apiKey,
    model: "gemini-1.5-flash",
    temperature: 0,
  });
  const prompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      "Answer the user's question based on the following context: {context}.",
    ],
    new MessagesPlaceholder("chat_history"),
    ["user", "{input}"],
  ]);
  const chain = await createStuffDocumentsChain({
    llm: model,
    prompt: prompt,
  });
  const retriever = vectorStore.asRetriever({
    k: 4,
  });
  /* As we have said before, we can pipe multiple chains at once or we can use a chain to create another chain, like this one.
  This chain takes in the history + the user's input + data from the vector storage. */
  const retrieverPrompt = ChatPromptTemplate.fromMessages([
    new MessagesPlaceholder("chat_history"),
    ["user", "{input}"],
    [
      "user",
      "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation",
    ],
  ]);
  /* We define a new retriever which takes in the old retriever and the new prompt. Making the LLM aware of the history. */
  const historyAwareRetriever = await createHistoryAwareRetriever({
    llm: model,
    retriever: retriever,
    rephrasePrompt: retrieverPrompt,
  });
  /* The new retrieval chain will take in the history awareness retriever to make it history + user's input + vector storage. */
  const retrievalChain = await createRetrievalChain({
    combineDocsChain: chain,
    retriever: historyAwareRetriever,
  });
  return retrievalChain;
};

// --------------------------
// Section: Implementations (using try...catch)
// --------------------------
try {
  // const vectorStore = await createVectorStore();
  // const chain = await createChain(vectorStore);
  // const response = await chain.invoke({
  //   input: "What is LCEL?",
  // });
  // console.log(response);
  // const vectorStore = await createVectorStore();
  // const chain = await createChainHistory(vectorStore);
  // const response = await chain.invoke({
  //   input: "How old is my dog? And what type of dog do I have?",
  //   chat_history: chatHistory,
  // });
  // console.log(response);
  const vectorStore = await createVectorStore();
  const chain = await createChainHistoryImprove(vectorStore);
  const response = await chain.invoke({
    input: "What is it?",
    chat_history: chatHistory,
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
  - Creating test chat history + Adding chat history -> Retrieve data from chat history.
  - Adding user input as data for retrieval, should use Langsmith or Langfuse to see the trace.
  Question:
*/
