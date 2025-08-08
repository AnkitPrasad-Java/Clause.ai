# AI Legal Clause Generator with Risk Classification

In this project we basically let the user input a prompt and we generate a clause specific to the user prompt through a static dataset, it also classifies the generated as low/medium/high risk(Is it feasible according to the industry standards).

# Dataset

The dataset which was used is CUAD v1(Contract Understanding Atticus Dataset), the dataset consists of multiple contracts of different comapanies and the contracts are of multiple types.
The contracts in the dataset are in .txt and .pdf formats.

# Flow
We created a vector database of the dataset through FAISS and applied RAG on the vector database and gemini to generate a clause but the generation of clause was done through a static data and we also wanted real time data so we applied Serper API to let gemini fetch real time data through the internet. The generated clause was classified on its type and risk.
We also added a feature called "Contract Summarizer" which allowed the user to input any legal contract as PDF file and the contract will be summarized in simpler terms and a Text-to-Speech was also applied on the summarized text.
