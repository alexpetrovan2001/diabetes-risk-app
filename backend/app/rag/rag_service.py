import ollama

from app.rag.retriever import OUT_OF_SCOPE_THRESHOLD, retrieve

OLLAMA_MODEL = "llama3.2:1b"

OUT_OF_SCOPE_MESSAGE = (
    "This question is outside the scope of this tool. "
    "This assistant only covers topics related to diabetes risk, including: "
    "blood glucose levels, BMI and body weight, blood pressure, insulin, "
    "diabetes types (Type 1, Type 2, gestational), risk factors, symptoms, "
    "diagnosis thresholds, prevention, and interpretation of prediction results. "
    "For other health questions, please consult a qualified healthcare professional."
)

SYSTEM_PROMPT = (
    "You are a medical information assistant for a diabetes risk education tool. "
    "Your role is to provide clear, accurate, and grounded explanations based only "
    "on the provided reference text. Do not invent information not present in the "
    "reference. Always remind the user that predictions are informational only and "
    "not a substitute for professional medical advice. Keep answers concise."
)


def _build_context(chunks: list[str]) -> str:
    return "\n\n---\n\n".join(chunks)


def explain_prediction(
    risk_label: str,
    probability: float,
    glucose: int,
    bmi: float,
    age: int,
    pregnancies: int,
    blood_pressure: int,
    skin_thickness: int,
    insulin: int,
    diabetes_pedigree_function: float,
) -> str:
    query = (
        f"Explain a diabetes risk prediction result. "
        f"Risk level: {risk_label}. Probability: {probability:.1%}. "
        f"Input values: glucose={glucose}, BMI={bmi}, age={age}, "
        f"pregnancies={pregnancies}, blood pressure={blood_pressure}, "
        f"skin thickness={skin_thickness}, insulin={insulin}, "
        f"diabetes pedigree function={diabetes_pedigree_function}. "
        f"What do these values mean for diabetes risk?"
    )

    chunks, best_distance = retrieve(query)
    if best_distance > OUT_OF_SCOPE_THRESHOLD:
        return OUT_OF_SCOPE_MESSAGE

    context = _build_context(chunks)

    prompt = (
        f"Question: {query}\n\n"
        f"Reference information:\n{context}\n\n"
        f"Instructions: Using only the reference information above, provide a short "
        f"explanation (3-5 sentences) of what this specific prediction result means. "
        f"Focus on the patient's actual input values and what they indicate about "
        f"diabetes risk. Do not give general information — address these specific values."
    )

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    return response["message"]["content"]


def answer_question(question: str) -> str:
    chunks, best_distance = retrieve(question)
    if best_distance > OUT_OF_SCOPE_THRESHOLD:
        return OUT_OF_SCOPE_MESSAGE

    context = _build_context(chunks)

    prompt = (
        f"Question: {question}\n\n"
        f"Reference information:\n{context}\n\n"
        f"Instructions: Answer the question above directly and specifically. "
        f"Start your answer by addressing exactly what was asked. "
        f"Use only the reference information provided. "
        f"Do not start with unrelated context. "
        f"If the answer is not in the reference, say so clearly."
    )

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    return response["message"]["content"]
