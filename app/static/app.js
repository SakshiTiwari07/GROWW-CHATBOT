const form = document.querySelector("#ask-form");
const questionInput = document.querySelector("#question");
const transcript = document.querySelector("#transcript");
const sendButton = document.querySelector("#send");
const exampleButtons = [...document.querySelectorAll("[data-question]")];

exampleButtons.forEach((button) => {
  button.addEventListener("click", () => {
    questionInput.value = button.dataset.question;
    form.requestSubmit();
  });
});

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) {
    return;
  }
  ask(question);
});

async function ask(question) {
  setBusy(true);
  const exchange = startExchange(question);
  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const payload = await response.json();
    if (!response.ok) {
      showError(exchange, failureMessage(payload));
      return;
    }
    showAnswer(exchange, payload);
  } catch (_error) {
    showError(exchange, "The question could not be answered.");
  } finally {
    setBusy(false);
  }
}

function startExchange(question) {
  const empty = transcript.querySelector(".empty");
  if (empty) {
    empty.remove();
  }
  const exchange = document.createElement("article");
  exchange.className = "exchange";
  const asked = document.createElement("p");
  asked.className = "question";
  asked.textContent = question;
  const status = document.createElement("p");
  status.className = "answer";
  status.textContent = "Looking up the loaded pages…";
  exchange.append(asked, status);
  transcript.append(exchange);
  exchange.scrollIntoView({ block: "nearest" });
  return exchange;
}

function showAnswer(exchange, payload) {
  const answer = exchange.querySelector(".answer");
  answer.textContent = answerText(payload.answer, payload.citation_url);
  const link = document.createElement("a");
  link.className = "citation";
  link.href = payload.citation_url;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  link.textContent = payload.citation_url;
  const updated = document.createElement("p");
  updated.className = "updated";
  updated.textContent = `Last updated from sources: ${payload.last_updated}`;
  exchange.append(link, updated);
}

function failureMessage(payload) {
  if (payload && payload.error && String(payload.error.code) === "401") {
    return "This link is private. In the Vercel project, turn off Deployment Protection, then open the site again.";
  }
  if (payload && typeof payload.detail === "string" && payload.detail) {
    return payload.detail;
  }
  if (payload && payload.error && payload.error.message) {
    return payload.error.message;
  }
  return "The question could not be answered.";
}

function showError(exchange, message) {
  exchange.querySelector(".answer").textContent = message;
}

function answerText(answer, citation) {
  return answer
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line && line !== citation && line !== `Source: ${citation}` && !line.startsWith("Last updated from sources:"))
    .join("\n");
}

function setBusy(busy) {
  sendButton.disabled = busy;
  exampleButtons.forEach((button) => {
    button.disabled = busy;
  });
  questionInput.disabled = busy;
}
