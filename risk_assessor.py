import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
# --- Setup ---
# Configure the API key (it's best to use environment variables)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


class RiskAssessor:
    """
    A class to assess and classify legal clauses using the Gemini API (Synchronous Version).
    """
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        # Define the possible classifications as a class attribute
        self.clause_categories = [
            "Termination", "Confidentiality", "Liability", "Payment", 
            "Force Majeure", "Intellectual Property", "Warranties", 
            "Governing Law", "Dispute Resolution", "Severability", "Assignment", 
            "Amendment", "Notices", "Privacy", "Compliance", "Indemnification", 
            "Non-Compete", "Data Protection", "Audit Rights", "Duration", 
            "Scope of Work", "Representations", "Escrow", "Arbitration", "General"
        ]

    def assess_risk(self, clause: str) -> str:
        """Assesses the risk of a clause using a synchronous API call."""
        prompt = f"""
        You are an expert legal risk analyst. Classify the risk of the following contract clause 
        as "High", "Medium", or "Low" based on its potential for financial, legal, or operational harm.

        Clause:
        ---
        {clause}
        ---

        Return ONLY one word: High, Medium, or Low.
        """
        try:
            # Use the synchronous `generate_content` method
            response = self.model.generate_content(prompt)
            risk_level = response.text.strip().capitalize()
            return risk_level if risk_level in ["High", "Medium", "Low"] else "Error: Unexpected risk response"
        except Exception as e:
            print(f"An error occurred during risk assessment: {e}")
            return "Error: API call failed"

    def classify_clause(self, clause: str) -> str:
        """Classifies a clause into a specific category using a synchronous API call."""
        categories_str = ", ".join(self.clause_categories)
        prompt = f"""
        You are an expert legal assistant. 
        Analyze the clause and return only the single most appropriate category name.

        Clause:
        ---
        {clause}
        ---

        Category:
        """
        try:
            # Use the synchronous `generate_content` method
            response = self.model.generate_content(prompt)
            classification = response.text.strip()
            return classification if classification in self.clause_categories else "General"
        except Exception as e:
            print(f"An error occurred during classification: {e}")
            return "Error: API call failed"

    def get_risk_level_value(self, risk: str) -> int:
        """Returns a numerical value for a given risk level."""
        risk_map = {"Low": 1, "Medium": 2, "High": 3}
        return risk_map.get(risk, 99)

# --- Example Usage (Synchronous) ---
def main():
    assessor = RiskAssessor()
    
    clause_example = "Upon project completion, all rights, title, and interest in the developed software shall be transferred to the Client."
    
    print(f"Analyzing Clause: '{clause_example}'\n")

    # The calls now happen one after the other (sequentially)
    classification = assessor.classify_clause_gemini(clause_example)
    risk = assessor.assess_risk_gemini(clause_example)
    
    risk_value = assessor.get_risk_level_value(risk)

    print(f"Clause Classification: {classification} 🤖")
    print(f"Assessed Risk: {risk} (Value: {risk_value}) ⚖️")

if __name__ == "__main__":
    if not os.getenv("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable not set.")
    else:
        main()