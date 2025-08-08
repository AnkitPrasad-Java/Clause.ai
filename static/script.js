// Import Firebase core and auth modules from CDN
import { initializeApp } from "https://www.gstatic.com/firebasejs/10.12.0/firebase-app.js";
import { getAuth, signInWithEmailAndPassword } from "https://www.gstatic.com/firebasejs/10.12.0/firebase-auth.js";

// Your Firebase config (this is fine to keep public in frontend)
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
const auth = getAuth(app); // 🧠 Authentication instance

// Handle login form submission
const form = document.getElementById("login-form");
const errorMessage = document.getElementById("error-message");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const email = form.email.value;
  const password = form.password.value;

  try {
    const userCredential = await signInWithEmailAndPassword(auth, email, password);
    alert("✅ Login successful! Redirecting...");
    window.location.href = "dashboard.html"; // redirect to dashboard
  } catch (error) {
    console.error(error.message);
    errorMessage.textContent = error.message;
  }
});