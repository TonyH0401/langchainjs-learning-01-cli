# LangChainJS Learning 01 CLI

Welcome to my LangChainJS repository implementation! This project is designed to provide a comprehensive guide to understanding and using LangChainJS, a powerful LLM Framework in a JavaScript library.

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Environment Variable](#Environment-Variable)
- [Quick Start](#Quick-Start)
- [Development Documentation](#development-documentation)

## Introduction

This is my implementation of LangChainJS - An LLM Framework in a JavaScript library. Each lesson has notes, synopsis and questions for future lessons to answer and expand on. There are also several extra/side projects I was researching included in the folder.

The original tutorial playlist that I based on is linked [here](https://youtube.com/playlist?list=PL4HikwTaYE0EG379sViZZ6QsFMjJ5Lfwj&si=o1vtvZ9lAB8sETbH). I will also have the playlist displayed in case the hyperlink is not accessible https://youtube.com/playlist?list=PL4HikwTaYE0EG379sViZZ6QsFMjJ5Lfwj&si=o1vtvZ9lAB8sETbH.

## Getting Started

I recommend running this project on **NodeJS v20+**. This project was originally running on **NodeJS v21.7.1**.

### Installation

To get started, you need to download this project from Github and navigate to the project's folder.

```sh
cd langchainjs-learning-01-cli/
```

Dowloading the project's dependencies.

```sh
npm install
npm install nodemon --save-dev
```

### Environment Variable

This step is **important**! Create an `.env` file to store your API KEY(s). These are the API KEY(s) you will need. Currently, in this project, I am using Google's Gemini API but you can change it to any LLM(s) you prefer.

```sh
GOOGLE_API_KEY=""
OPENAI_API_KEY=""
TAVILY_API_KEY=""
LANGFUSE_PUB=""
LANGFUSE_SEC=""
```

## Quick Start

Run the project using the following command(s).

```sh
node .\src\<file name>.js
```

## Development Documentation

Order by newest to oldest.

### 31/07/2024

- There was a test file on the OpenAI LLM, the folder was called `legacyOpenAI.js`, I deleted it because there was no use, nothing was working because nothing is free for OpenAI anymore.

### 12/06/2024

- Should change in `package.json` to `"type":"module"` first before downloading any npm packges. If you don't do this, you need to delete `node_modules/`, `package-lock.json` and wiping it clean. I remember reading this somewhere, forgot to noted it down.
