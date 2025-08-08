// signup.js
import { initializeApp } from "https://www.gstatic.com/firebasejs/10.12.0/firebase-app.js";
import {
  getAuth,
  createUserWithEmailAndPassword
} from "https://www.gstatic.com/firebasejs/10.12.0/firebase-auth.js";

// ✅ Your actual Firebase config:
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

// Handle signup form
const form = document.getElementById("signup-form");
const errorMessage = document.getElementById("error-message");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const email = form.email.value;
  const password = form.password.value;

  try {
    const userCredential = await createUserWithEmailAndPassword(auth, email, password);
    alert("✅ Signup successful! Redirecting to login...");
    window.location.href = "index.html"; // or dashboard.html if you want
  } catch (error) {
    console.error(error.message);
    errorMessage.textContent = error.message;
  }
});