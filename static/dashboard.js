// Import Firebase core and auth modules from CDN
import { initializeApp } from "https://www.gstatic.com/firebasejs/10.12.0/firebase-app.js";
import { getAuth, onAuthStateChanged, signOut } from "https://www.gstatic.com/firebasejs/10.12.0/firebase-auth.js";

// Your Firebase config
const firebaseConfig = {
  apiKey: "AIzaSyDyt3r9AGcHkuvL37frfaKFzr9er18OGMI",
  authDomain: "clause-9d753.firebaseapp.com",
  projectId: "clause-9d753",
  storageBucket: "clause-9d753.firebasestorage.app",
  messagingSenderId: "490035572199",
  appId: "1:490035572199:web:531332d89e2a2e690ba142",
  measurementId: "G-DMWH6P65EW"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);

// UI Elements
const userNameEl = document.getElementById('user-name');
const userEmailEl = document.getElementById('user-email');
const logoutButton = document.getElementById("logout");

const evaluateButton = document.getElementById("evaluate");
const regenerateButton = document.getElementById("regenerate");
const promptInput = document.getElementById("prompt");
const resultsDiv = document.getElementById("results");
const promptList = document.getElementById('promptList');

const pdfUploadInput = document.getElementById('pdf-upload');
const summarizePdfBtn = document.getElementById('summarize-pdf-btn');
const summaryOutputDiv = document.getElementById('summary-output');

let lastPrompt = '';

// --- Authentication Logic ---

// Handle logout
logoutButton.addEventListener("click", async () => {
    try {
        await signOut(auth);
        // Redirect to login page after logout
        window.location.href = "index.html";
    } catch (error) {
        console.error("Logout error:", error.message);
        alert("Logout failed: " + error.message);
    }
});

// Authentication state observer
onAuthStateChanged(auth, (user) => {
    if (user) {
        // User is signed in
        const email = user.email;
        const displayName = user.displayName || email.split('@')[0];

        userNameEl.textContent = displayName;
        userEmailEl.textContent = email;
    } else {
        // Not logged in -> redirect to login page
        window.location.href = 'index.html';
    }
});

// --- Utility Functions ---
function getRiskBadgeClass(risk) {
    switch(risk.toLowerCase()) {
        case 'low': return 'risk-low';
        case 'medium': return 'risk-medium';
        case 'high': return 'risk-high';
        case 'very high': return 'risk-very-high';
        default: return 'risk-medium';
    }
}

function showLoadingState() {
    resultsDiv.innerHTML = `
        <div class="flex items-center justify-center py-12">
            <div class="text-center">
                <div class="w-12 h-12 mx-auto mb-4 loading-spinner">
                    <svg class="w-full h-full text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
                    </svg>
                </div>
                <p class="text-lg font-medium text-gray-700">Generating and assessing your clause...</p>
                <p class="text-sm text-gray-500 mt-2">This may take a few moments</p>
            </div>
        </div>
    `;
}

async function downloadPdf(clauseText) {
    try {
        const response = await fetch("http://127.0.0.1:8000/download_pdf", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ clause: clauseText }),
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || "PDF generation failed.");
        }

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "generated_clause.pdf";
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);

    } catch (error) {
        console.error("Download PDF error:", error);
        alert(`Error downloading PDF: ${error.message}`);
    }
}

async function getEvaluation(prompt) {
    showLoadingState();
    regenerateButton.style.display = "none";

    try {
        const user = auth.currentUser;
        if (!user) {
            resultsDiv.innerHTML = `
                <div class="text-center py-12">
                    <p class="text-red-500 font-medium">Please log in to use this feature.</p>
                </div>
            `;
            return;
        }
        const idToken = await user.getIdToken();

        const response = await fetch("http://127.0.0.1:8000/evaluate", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${idToken}`
            },
            body: JSON.stringify({ prompt })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || "An unknown error occurred.");
        }

        const data = await response.json();
        const riskClass = getRiskBadgeClass(data.risk);
        
        resultsDiv.innerHTML = `
            <div class="result-card rounded-xl p-6">
                <div class="mb-6">
                    <div class="flex items-center justify-between mb-4">
                        <h4 class="text-lg font-semibold text-gray-800">Generated Clause</h4>
                        <span class="risk-badge ${riskClass}">${data.risk} Risk</span>
                    </div>
                    <div class="bg-gray-50 rounded-lg p-4 border-l-4 border-indigo-500">
                        <p class="text-gray-800 leading-relaxed">${data.clause.replace(/\*/g, '')}</p>
                    </div>
                </div>
                
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                    <div class="bg-blue-50 rounded-lg p-4">
                        <h5 class="font-semibold text-blue-800 mb-2">Classification</h5>
                        <p class="text-blue-700">${data.classification}</p>
                    </div>
                    <div class="bg-purple-50 rounded-lg p-4">
                        <h5 class="font-semibold text-purple-800 mb-2">Source</h5>
                        <p class="text-purple-700">${data.source}</p>
                    </div>
                </div>
                
                <div class="flex space-x-3">
                    <button id="download-btn" class="px-6 py-2 bg-green-600 hover:bg-green-700 text-white font-semibold rounded-lg transition duration-300 shadow-md hover:shadow-lg">
                        <svg class="w-4 h-4 inline mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
                        </svg>
                        Download PDF
                    </button>
                </div>
            </div>
        `;
        
        regenerateButton.style.display = "inline-block";

        // Add to sidebar
        const li = document.createElement('li');
        li.className = 'sidebar-item p-3 rounded-lg bg-gray-50 hover:bg-indigo-50 transition-all duration-300';
        li.innerHTML = `
            <div class="text-sm font-medium text-gray-800">${prompt.substring(0, 50)}${prompt.length > 50 ? '...' : ''}</div>
            <div class="text-xs text-gray-500 mt-1">Just now</div>
        `;
        promptList.prepend(li);

        document.getElementById('download-btn').addEventListener('click', (event) => {
            event.preventDefault();
            downloadPdf(data.clause);
        });

    } catch (error) {
        console.error("Evaluation error:", error);
        resultsDiv.innerHTML = `
            <div class="text-center py-12">
                <div class="w-16 h-16 mx-auto mb-4 bg-red-100 rounded-full flex items-center justify-center">
                    <svg class="w-8 h-8 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                    </svg>
                </div>
                <p class="text-red-500 font-medium text-lg">Error occurred</p>
                <p class="text-gray-600 mt-2">${error.message}</p>
            </div>
        `;
    }
}

// --- Event Listeners for Clause Generation ---
evaluateButton.addEventListener("click", async () => {
    const prompt = promptInput.value.trim();
    if (!prompt) {
        alert("Please enter your requirements first.");
        return;
    }
    lastPrompt = prompt;
    getEvaluation(prompt);
});

regenerateButton.addEventListener("click", async () => {
    if (lastPrompt) {
        getEvaluation(lastPrompt);
    } else {
        resultsDiv.innerHTML = `
            <div class="text-center py-12">
                <p class="text-orange-500 font-medium">No previous prompt to regenerate.</p>
            </div>
        `;
    }
});

// Add enter key support for textarea
promptInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && e.ctrlKey) {
        evaluateButton.click();
    }
});

// --- PDF Summarizer Logic ---
summarizePdfBtn.addEventListener('click', async (e) => {
    e.preventDefault();
    
    const file = pdfUploadInput.files[0];
    if (!file) {
        alert("Please select a PDF file to summarize.");
        return;
    }

    if (file.type !== 'application/pdf') {
        alert("Only PDF files are supported.");
        return;
    }

    // Show loading state
    summaryOutputDiv.innerHTML = `
        <div class="flex items-center justify-center py-6">
            <div class="w-8 h-8 mx-auto loading-spinner">
                <svg class="w-full h-full text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
                </svg>
            </div>
            <p class="text-gray-600 ml-3">Summarizing PDF...</p>
        </div>
    `;
    
    // Disable the button to prevent multiple clicks
    summarizePdfBtn.disabled = true;
    const originalButtonText = summarizePdfBtn.innerHTML;
    summarizePdfBtn.innerHTML = `
        <svg class="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg>
        Processing...
    `;

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch("http://127.0.0.1:8000/summarize_pdf", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            let errorMessage = "PDF summarization failed.";
            try {
                const errorData = await response.json();
                errorMessage = errorData.detail || errorMessage;
            } catch (e) {
                errorMessage = response.statusText || errorMessage;
            }
            throw new Error(errorMessage);
        }

        const data = await response.json();
        const summaryText = data.summary;
        
        summaryOutputDiv.innerHTML = `
            <div class="bg-gray-50 rounded-lg p-4 border-l-4 border-purple-500">
                <h5 class="font-semibold text-gray-800 mb-2">Summary:</h5>
                <p class="text-gray-800 leading-relaxed">${summaryText}</p>
            </div>
        `;

    } catch (error) {
        console.error("PDF Summarization error:", error);
        summaryOutputDiv.innerHTML = `
            <div class="text-center py-4">
                <p class="text-red-500 font-medium">Error: ${error.message}</p>
            </div>
        `;
    } finally {
        summarizePdfBtn.disabled = false;
        summarizePdfBtn.innerHTML = originalButtonText;
    }
});