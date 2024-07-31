// --------------------------
// Section: Import Libraries
// --------------------------
import * as dotenv from "dotenv";
dotenv.config();
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";

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
  // verbose: true,
});

// --------------------------
// Section: Create a string output parser
// --------------------------
const outputParser = new StringOutputParser();

// --------------------------
// Section: Generate MySQL Queries Process
// --------------------------
const schema = `
  CREATE TABLE dbo.Student(
    student_id varchar(10) primary key,
    first_name nvarchar(10),
    last_name nvarchar(10),
    city nvarchar(64),
    age int
  )
`;

const prompt = ChatPromptTemplate.fromTemplate(`
  You are a MySQL expert.

  Your goal is to generate syntax correct MySQL "SELECT" queries based on the given MySQL schemas.
  You need to follow these rules and guidelines. No yapping.

  Rules and Guidelines:
  - DO:
    -- Displays from minumum 3 to maximum 5 properties in the "SELECT" queries. 
    -- The schema's primary key(s) must always be used in "SELECT" queries.
    -- If there are tables need to be joined, always use 'JOIN' to join tables.
    -- Always use 'LIMIT' to limit the out to 20 rows.
  - DO NOT:
    -- Use '*' when generating "SELECT" queries.

  Given the MySQL database schema: {schema}.
  Generate MySQL "SELECT" query based on the following user input: {user_input}.
  Display the following column(s): {column}.
  `);
/* Create (Base) Chain */
const chain = prompt.pipe(model).pipe(outputParser);

const response = await chain.invoke({
  schema: schema,
  user_input: "Find students who have age above 10",
  column: "student_id, first_name, last_name, age",
});

try {
  console.log(response);
} catch (error) {
  console.error(error.message);
}

// --------------------------
// Section: Synopsis
// --------------------------
/* 
  - combine with the tutorial video to get data from the user input specifications -> devide and process the input,
  they keyword here is "extract the information"
  - There is a function in Langchain for SQL but it has a default prompt already.
*/
