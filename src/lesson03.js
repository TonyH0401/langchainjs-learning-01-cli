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
// Section: Formatting Reponse (I put them into functions)
// --------------------------
/* String Output Parser (returns string) */
async function callStringOutputParser() {
  /* Create a prompt */
  const prompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      `You are a clever and funny comedian.
      Your jokes are short but meaningful.
      Tell a joke based on the following word provided by the user.`,
    ],
    ["human", "{input}"],
  ]);
  /* Create an output parser */
  const parser = new StringOutputParser();
  /* Creates a chain and chains (pipe) the output parser */
  /* The output of prompt becomes input for model, 
  the output of model becomes input for parser, 
  finally the result */
  const chain = prompt.pipe(model).pipe(parser);
  /* Invoke the chain, this is the final step */
  return await chain.invoke({
    input: "dog",
  });
}

/* Call List Output Parser (returns as list) */
async function callListOutputParser() {
  /* Create a prompt */
  const prompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      `You are a dictonary. 
      Provide 5 synonyms, seperated by commas, for the following word provided by the user.`,
    ],
    ["human", "{word}"],
  ]);
  /* Create an output parser that seperate the result by comma */
  const outputParser = new CommaSeparatedListOutputParser();
  /* Create a chain and pipe that chain through an output parser (chaining them together) */
  const chain = prompt.pipe(model).pipe(outputParser);
  /* Invoke the chain */
  return await chain.invoke({
    word: "happy",
  });
}

/* Object Output Parser (returns as an object) */
async function callStructuredOutputParser() {
  /* Create a prompt */
  /* Langchain prompt layout is in bulk, I prefer divide them into separate section (but I don't like it too separate).
  So, in this prompt I will divide the prompt into separate sections. */
  const prompt = ChatPromptTemplate.fromTemplate(`
    You are an information extracting expert. Your goal is to extract information correctly.

    Extract information from the following phrase. 
    If there is no information about a person's property, say "null". DO NOT make up information. No yapping.
    
    Formating instruction: {format_instruction}
    Phrase: {phrase}
    `);
  /* Define the Object Output Parser */
  /* The output parser's job is to reformat the the response from the LLM. Normally, it returns in string format.
  But with this, we config it to return in object format. The instructions we give it are as followed:
  - Reformat instructions: the object structure.
  - Data selection: categorizing and selecting the correct data from the user's input. 
 */
  const outputParser = StructuredOutputParser.fromNamesAndDescriptions({
    name: "the name of the person",
    age: "the age of the person",
    gender: "the gender of the person",
    occupation: "the occupation of the person",
  });
  /* Create a chain with the output parser */
  const chain = prompt.pipe(model).pipe(outputParser);
  /* Invoke the chain */
  return await chain.invoke({
    // phrase: "A 30 year-old boy from Sweden named Felix",
    phrase:
      "A 32 year-old male Youtuber from Sweden called Felix celebrating his channel reaching 100 million subscribers",
    format_instruction: outputParser.getFormatInstructions(),
  });
}

/* Object Output Parser using Zod (returns as an object) */
async function callZodOutputParser() {
  /* Create a prompt */
  const prompt = ChatPromptTemplate.fromTemplate(`
    You are an information extracting expert. Your goal is to extract information correctly.

    Extract information from the following phrase.
    If there is no information about a person's property, say "null". DO NOT make up information. No yapping.
    
    Formating instruction: {format_instruction}
    Phrase: {phrase}
    `);
  /* Define a zod schema to structure the object */
  const outputParser = StructuredOutputParser.fromZodSchema(
    z.object({
      recipe: z.string().describe("the recipe name"),
      ingredients: z
        .array(
          z.object({
            ingredient: z.string().describe("ingredient name"),
            amount: z.number().describe("ingredient amount"),
            measure: z.string().describe("ingredient measurement"),
            factory: z.string().describe("ingredient producer"),
          })
        )
        .describe(
          "an array of object contains ingredients and ingredients' amount"
        ),
    })
  );
  const chain = prompt.pipe(model).pipe(outputParser);
  return await chain.invoke({
    phrase:
      "You will need 100 grams flour from Norway, 200 litters water and 50 yeat to make bread.",
    format_instruction: outputParser.getFormatInstructions(),
  });
}

// --------------------------
// Section: Implementations (using try...catch)
// --------------------------
try {
  // const response = await callStringOutputParser();
  // console.log(response);
  // const response = await callListOutputParser();
  // console.log(response);
  // const response = await callStructuredOutputParser();
  // console.log(response);
  const response = await callZodOutputParser();
  console.log(response);
  // const response = await callZodOutputParser();
  // console.log(response);
} catch (error) {
  console.error(error.message);
}

// --------------------------
// Section: Synopsis
// --------------------------
/* 
  Learn:
  - We learn how to pipe from prompt to model to output parser to get our final result. This will help us in learning how to pipe through
  multiple chains.
  - How to structure the prompt.
  - How to structure the reponse from the LLM.
  Question:
*/
