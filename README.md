

I have successfully built the LexRetrieve AI system, including a full Python hybrid retrieval pipeline (BM25 + FAISS + reranking) and a production-ready implementation. It also includes a React-based interactive dashboard that simulates the end-to-end legal retrieval process in real time with exportable code for deployment.


<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# Run and deploy your AI Studio app

This contains everything you need to run your app locally.

View your app in AI Studio: https://ai.studio/apps/109322ec-5cc2-412f-9068-b1a45d296fb4

## Run Locally

**Prerequisites:**  Node.js


1. Install dependencies:
   `npm install`
2. Set the `GEMINI_API_KEY` in [.env.local](.env.local) to your Gemini API key
3. Run the app:
   `npm run dev`
