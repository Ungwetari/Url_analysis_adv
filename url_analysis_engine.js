import OpenAI from "openai";
import fetch from "node-fetch"; // Importing the node-fetch library to perform HTTP requests.

import * as cheerio from "cheerio"; // Importing the cheerio library for parsing and manipulating HTML content.

import dotenv from "dotenv";

dotenv.config(); // Loading environment variables from a .env file using dotenv.

const HF_API_TOKEN = process.env.HF_API_TOKEN; // Hugging Face API token to authorize requests to the Hugging Face Inference API.

const YOUTUBE_API_KEY = process.env.YOUTUBE_API_KEY; // YouTube API key, retrieved from environment variables.

const HF_API_OPENAI = process.env.HF_API_OPENAI;


const API_URL = // API endpoint for performing text classification with the Hugging Face model.
    "https://api-inference.huggingface.co/models/facebook/bart-large-mnli";

const YOUTUBE_API_URL = "https://www.googleapis.com/youtube/v3/videos"; // API endpoint for retrieving YouTube video details.



let interestPercentagesString = ""; // Initialize an empty string to collect all percentages.
import readline from "readline";
let username = "";
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

let urlsToAnalyze = [];
rl.question("Enter your username:\n", (name) => {
  username = name.trim();
  rl.question("Please enter URLs separated by 'SPACE':\n", (input) => {
    urlsToAnalyze = input.split(" ").filter((url) => url.trim() !== "");
    rl.close();
    analyzeURLs();
  });
})

// Function to extract YouTube video ID from a given URL

function extractYouTubeVideoID(url) { // Extracts the YouTube video ID from either the standard YouTube URL or shortened youtu.be link.
  const urlObj = new URL(url);
  if (urlObj.hostname === "youtu.be") {
    return urlObj.pathname.substring(1);
  }
  return urlObj.searchParams.get("v");
}

// Function to convert ISO 8601 duration strings (e.g., "PT10M30S") to total seconds

function parseISODuration(duration) { // Converts ISO duration format (used by YouTube API) into seconds for easier calculations.
  const regex = /PT(\d+H)?(\d+M)?(\d+S)?/;
  const parts = duration.match(regex);

  const hours = parts[1] ? parseInt(parts[1].replace("H", "")) : 0;
  const minutes = parts[2] ? parseInt(parts[2].replace("M", "")) : 0;
  const seconds = parts[3] ? parseInt(parts[3].replace("S", "")) : 0;

  return hours * 3600 + minutes * 60 + seconds;
}

// Function to retrieve the duration of a YouTube video using its video ID and the YouTube API

async function fetchYouTubeVideoDuration(videoId) { // Fetches the video duration for a given YouTube video ID.
  try {
    const url = `${YOUTUBE_API_URL}?part=contentDetails&id=${videoId}&key=${YOUTUBE_API_KEY}`;
    const response = await fetch(url);

    const data = await response.json();

    if (data.items && data.items.length > 0) {

      return parseISODuration(data.items[0].contentDetails.duration);
    }
    return null;
  } catch (error) {

    return null;
  }
}

// Function to compute the weight multiplier based on the length of a YouTube video

async function calculateLengthBasedWeight(url) { // Determines a multiplier for text analysis weights based on video length.
  const videoId = extractYouTubeVideoID(url);
  if (!videoId) return 1; // Neutral multiplier for non-YouTube URLs

  const durationInSeconds = await fetchYouTubeVideoDuration(videoId);
  if (!durationInSeconds) return 1;


  const durationInMinutes = durationInSeconds / 60;
  const multiplier = durationInMinutes > 60 ? 3 : durationInMinutes > 10 ? 2 : durationInMinutes > 0 ? 1.5 : 1;
  return multiplier;

}

// Function to retrieve and sanitize text content from given non-YouTube URLs

async function fetchURLContent(url) { // Retrieves text-based webpage content while excluding unnecessary elements like scripts/styles.
  if (url.includes("youtube.com") || url.includes("youtu.be")) {
    return await fetchYouTubeContent(url); // Handle YouTube URLs
  }
  try {
    const response = await fetch(url);
    const html = await response.text();
    const $ = cheerio.load(html);
    $("script, style, meta, link, noscript").remove();
    const bodyText = $("body").text().replace(/\s+/g, " ").trim();
    return bodyText.length > 8000 ? bodyText.substring(0, 8000) : bodyText;
  } catch (error) {
    console.error(`Error fetching ${url}:`, error);
    return null;
  }
}

// Function to retrieve metadata (e.g., title, description) for YouTube videos

async function fetchYouTubeContent(url) { // Extracts YouTube video metadata from the provided URL.
  try {
    const response = await fetch(url);
    const html = await response.text();
    const $ = cheerio.load(html);

    const title = $("meta[name='title']").attr("content") || $("title").text();
    const description = $("meta[name='description']").attr("content") || "";
    return `${title}. ${description}`.trim();
  } catch (error) {
    console.error(`Error fetching YouTube content from ${url}:`, error);
    return null;
  }
}






// Main function to process and classify each URL's content using the Hugging Face API

async function analyzeURLs() { // Orchestrates URL analysis, from content retrieval to text classification.
  try {
    const urlContents = await Promise.all(
        urlsToAnalyze.map(async (url) => {
          if (url.includes("youtube.com") || url.includes("youtu.be")) {
            const content = await fetchURLContent(url);
            const weight = await calculateLengthBasedWeight(url);
            const durationInSeconds = await fetchYouTubeVideoDuration(extractYouTubeVideoID(url))
            return {url, content, weight, durationInSeconds};
          } else {
            const content = await fetchURLContent(url);
            const weight = await calculateLengthBasedWeight(url);
            return {url, content, weight};
          }

        })
    );

    const validURLContents = urlContents.filter(({content}) => content !== null);

    const results = await Promise.all(
        validURLContents.map(async ({url, content, weight, durationInSeconds}) => {
          const response = await fetch(API_URL, {
            method: "POST",
            headers: {
              Authorization: `Bearer ${HF_API_TOKEN}`,
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              inputs: content,
              parameters: {
                candidate_labels: [
                  "Technology",
                  "News",
                  "Video Games",
                  "Entertainment",
                  "Science",
                  "Business",
                ],
              },
            }),
          });

          const result = await response.json();
          return {url, labels: result.labels, scores: result.scores, weight, durationInSeconds};
        })
    );

    console.log("\n" + `Hello ${username}, here are your URL Analysis Results:`);
    console.log("-------------------------------");

    let totalScores = {}; // Object for accumulating weighted scores for each classification label.

    let weightSum = 0; // Tracks the total weight multipliers for the analyzed URLs.

    results.forEach(({url, labels, scores, weight, durationInSeconds}) => {

      console.log("\n" + `Results received for ${url}:` + "\n"); // Logs the classification results for the URL.
      const formattedResults = labels
          .map((label, index) => `${label}: ${(scores[index] * 100).toFixed(2)}%`)
          .join(" | ");
      const videoInfo = ` | Multiplier: ${weight}x${(url.includes("youtube.com") || url.includes("youtube")) && durationInSeconds ? ` (Video Detected , length: ${Math.round(durationInSeconds / 60)} minutes)` : ""}`;

      console.log(`${formattedResults}${videoInfo}`);

      labels.forEach((label, idx) => {
        if (!totalScores[label]) totalScores[label] = 0;
        totalScores[label] += scores[idx] * weight; // Add weighted score
      });

      weightSum += weight;
    });

    console.log("-------------------------------");
    console.log(`User Engagement Profile for ${username} (Normalized Percentages):`);
    console.log("-------------------------------");

    const totalScoreSum = Object.values(totalScores).reduce((sum, score) => sum + score, 0); // Sum of all weighted classification scores for normalization.


    Object.entries(totalScores)
        .sort(([, a], [, b]) => b - a)
        .forEach(([label, score], index, array) => {
          const normalizedPercentage = ((score / totalScoreSum) * 100).toFixed(2); // Normalize
          console.log(` - ${label}: ${normalizedPercentage}%`);
          // Add to the interestPercentagesString
          interestPercentagesString += `${label}: ${normalizedPercentage}%`;
          // Append a comma and space, except for the last entry
          if (index !== array.length - 1) {
            interestPercentagesString += ", ";
          }
        });

    console.log("-------------------------------" + "\n" + "Thank you for generating this Engagement profile , ChatGPT suggests the following products:"+"\n")


  } catch (error) {
    console.error("Error analyzing URLs:", error);
  }



  const openai = new OpenAI({
    apiKey: HF_API_OPENAI,
  });

  const response = openai.chat.completions.create({
    model: "gpt-4o-mini",
    store: true,
    messages: [
      {"role": "user", "content": `Suggest five prodcuts focused on the top two interest percentages:${interestPercentagesString} , just name them no explanation`},
    ],
  });

  response.then((result) => console.log(result.choices[0].message.content));



}