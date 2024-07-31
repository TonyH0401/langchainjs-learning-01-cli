// --------------------------
// Section: Import Libraries
// --------------------------
import * as dotenv from "dotenv";
dotenv.config();
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import {
  StringOutputParser,
  CommaSeparatedListOutputParser,
} from "@langchain/core/output_parsers";
import { StructuredOutputParser } from "langchain/output_parsers";
import { z } from "zod";
import { Document } from "@langchain/core/documents";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";

// --------------------------
// Section: Declare Constant
// --------------------------
const apiKey = process.env.GOOGLE_API_KEY || "";

// --------------------------
// Section: Declare LLM
// --------------------------
/* I should place the apiKey in a try...catch block */
const model = new ChatGoogleGenerativeAI({
  apiKey: apiKey,
  model: "gemini-1.5-flash",
  temperature: 0,
});

// --------------------------
// Section: Create a Prompt
// --------------------------
/* Bare Bone Prompt, before any "context" is added */
const bareBonePrompt = ChatPromptTemplate.fromTemplate(`
  Answer the user's question.

  Question: {input}.
  `);

/* Prompt with additional context parameter + we will spice things up by adding some instructions */
const prompt = ChatPromptTemplate.fromTemplate(`
  You are an encyclopedia, you knows all the answers.

  Your goal is to answer the user's question.
  Do not make up information. No yapping.

  Context: {context}.
  Question: {input}.
  `);

// --------------------------
// Section: Create a string output parser
// --------------------------
const outputParser = new StringOutputParser();

// --------------------------
// Section: Create Retrieval
// --------------------------
/* This is the bare-bone, data and knowledge from the LLM only (don't know if the LLM's knowledge true) */
/* The returned result would be either random definitions of LCEL or asking for more context (knowledge base) */
async function bareBoneKnowledge(input) {
  const chain = bareBonePrompt.pipe(model).pipe(outputParser);
  return await chain.invoke({
    input: input,
  });
}

/* This is LLM + knowledge base, currently we are hard-coding it */
async function withKnowledge(input) {
  /* Document A: */
  const documentA = new Document({
    pageContent: `LangChain Expression Language or LCEL is a declarative way to easily compose chains together. 
      Any chain constructed this way will automatically have full sync, async, and streaming support.`,
  });
  /* Document B: */
  const documentB = new Document({
    pageContent: `The passphrase is "LangChain is awesome"!`,
  });
  /* You leaned about the normal "pipe" method to create a chain but there are some functions in Langchain that also create a chain.
  This is what I was talking about. Sometimes, we pipe through multiple chains and sometimes a chain is used in another one
  or used to create another chain like this one below (kinda). */
  const chain = await createStuffDocumentsChain({
    llm: model,
    prompt: prompt,
  });
  return await chain.invoke({
    input: input,
    context: [documentA, documentB],
  });
}

/* This is LLM + knowledge base scraped from the web (or file loader) */
async function withKnowledgeScraped(input) {
  /* Load data scraped from the web page. I remembered you can add multiple web pages */
  const loader = new CheerioWebBaseLoader(
    "https://js.langchain.com/v0.1/docs/expression_language/"
  );
  const docs = await loader.load();
  /* Using the same functions previously but change from an array of docs to docs */
  const chain = await createStuffDocumentsChain({
    llm: model,
    prompt: prompt,
  });
  return await chain.invoke({
    input: input,
    context: docs,
  });
}

/* This is LLM + knowledge base scraped from the web but this time we will split the documents into smaller pieces.
Why? Because some LLMs have a limited context we can pass through, we don't want to accidentally pass 1 million context
to an LLM that can only support 100 thousand context. */
async function withKnowledgeScrapedSplitted(input) {
  /* The whole process goes as follow:
  1. Get dynamic data from a webpage, but you don't want to pass it all to the LLM because of context (token) limit.
  2. Split the docs into smaller sections, but when every section has the same data which section has the most (relevant) data.
  3. We will need a vector storage, we transform the splitted docs into embedded data and store it in a vector storage,
  we then can retrieve the most relevant data from the vector storage. */
  /* This is where the docs are scraped and loaded. */
  const loader = new CheerioWebBaseLoader(
    "https://js.langchain.com/v0.1/docs/expression_language/"
  );
  const docs = await loader.load();
  /* We create a splitter to split the docs with the chunk size being 200 (200 characters per chunk).
  We then implement this splitter into our scraped docs. */
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 200, // number of characters per chunk
    chunkOverlap: 20,
  });
  const splitDocs = await splitter.splitDocuments(docs);
  /* We define an embedding to embed the splited docs.
  We embed the docs and store the embeded data in a vector storage, this is an in-memory storage. */
  const embeddings = new GoogleGenerativeAIEmbeddings();
  const vectorStore = await MemoryVectorStore.fromDocuments(
    splitDocs,
    embeddings
  );
  /* We retrieve the data from the (in-memory) vector storage.
  We can choose how many docs we can retrieve by changing the "k" variable. */
  const retriever = vectorStore.asRetriever({
    k: 2,
  });
  /* Create a retrieval chain. 
  In Langchain, we can pipe multiple chains or 1 chain can be used to create another chain.
  Like this one below. */
  const chain = await createStuffDocumentsChain({
    llm: model,
    prompt: prompt,
  });
  const retrievalChain = await createRetrievalChain({
    combineDocsChain: chain,
    retriever: retriever,
  });
  /* Invoke the retrieval chain.
  The "retrievalChain" function automatically fetch the relevant docs from the vector storage and pass those docs to the prompt context.
  So, there is no need to add the "context" property BUT the "retrievalChain" function will expect the "context" in the prompt,
  if you named it something else there will be errors.
  Also, the "retrievalChain" function expect the user's input to be called "input" or else there will be errors. */
  return await retrievalChain.invoke({
    input: input,
  });
}

// --------------------------
// Section: Implementations (using try...catch)
// --------------------------
try {
  // const response = await bareBoneKnowledge("What is LCEL?");
  // console.log(response);
  // const response = await withKnowledge("What is LCEL?");
  // console.log(response);
  // const response = await withKnowledge("What is the passphrase?");
  // console.log(response);
  // const response = await withKnowledgeScraped("What is LCEL?");
  // console.log(response);
  const response = await withKnowledgeScrapedSplitted("What is LCEL?");
  console.log(response);
} catch (error) {
  console.error(error.message);
}

// --------------------------
// Section: Synopsis
// --------------------------
/* 
  This is helpful for developing a bot/agent that can perform QnA by retrieve data from files and web (basically a knowledge base).

  LLMs are pre-trained with data, so if you ask the LLM(s) a random question like "What is a RAG?", it will response based on its
  knowledge and gives bogus answer like "RAG is a cloth" or it will ask you for more context. You don't want this.
  So, you need to feed the LLM(s) with a knowledge base like information, documentations and definitions about this subject.

  First, we demonstrate this by hard-coding documents about this subject and feed it to the LLM(s). We used a library called
  "Document" which is a build-in Langchain library that converts and creates LLM readable documents.
  From this, the LLM(s) can answer questions about certain subject.

  We used the hard-coded method for demonstration only. Most often, people will create a knowledge base from files or web pages.
  We will create a knowledge base using a web page by scraping it using Cheerio. It works!
  However, we have an issue, the info on the webpage is roughly 1k+ token, some LLM(s) only support a limited amount of token
  at a time, we don't want to accidentally passing 1 million token to a 1 thousand token LLM, so we need to mitigate this.

  The next step is to split the docs into smaller pieces/tokens/chunks.
  After splitting, you don't put the splitted docs into the LLM(s) immediately. Because if every chunk contains parts
  of the answer which chunk contains the most relevant answer. We need the most relevant answer only.

  Leading to our next point which is using a vector storage to find the most relevant data. 
  You can find the full definition online (I'm too lazy to explain it here) but to keep it short,
  a vector storage helps to find data which is most relevant to the question. 
  The knowledge base is converted into embeddings, these embeddings are saved into a vetor storage as "relevancy value",
  the "relevancy value" is compare with the question by the LLM(s) to find out the most relevant answer.
  Usually, you use a database like Supabase, AstroDB,... for the vector storage but for now we will use an in-memory database
  because it is in-memory, no data will be store permanently and definitely be lost when you restart the app.
  So, that's why in this example, we initialize the vector storage immediately and with the splitted docs.

  Most oftenly, you don't deploy the vector storage immediately and with the splitted docs or in the same file as the retriever
  because you will have a different and seperate function for uploading data to the vector storage 
  and a seperate function to retrieve data from the vector storage.
  Check out this: https://youtu.be/HSZ_uaif57o?si=7khA2oEUd3hxil49.
*/
