window.onload = () => {
  // TEST_CODE is in data.js.
  loadCode(TEST_CODE);
};

function loadCode(codeText) {
  let codebox = document.getElementById("da-code");
  codebox.innerHTML = "";
  loadSidebar(DATA);

  let lineToQuestionCount = TEST_CODE.split("\n").map((_, idx) => {
    let lineNum = idx + 1;
    return DATA.Questions.filter((q) => {
      let [lo, hi] = q.Lines;
      return lo <= lineNum && lineNum <= hi;
    }).length;
  });
  let maxQuestionCount = Math.max(...lineToQuestionCount);

  // For each line of code
  TEST_CODE.split("\n").forEach((line, idx) => {
    idx++;

    let uberSpan = document.createElement("span");
    uberSpan.id = `L${idx}`;
    uberSpan.classList.add("loc-span");
    codebox.append(uberSpan);

    let lineNumSpan = document.createElement("span");
    lineNumSpan.innerHTML = String("   " + idx).slice(-3) + "  ";
    // Calculate the highlight :)
    lineNumSpan.style.backgroundColor = `rgba(205, 0, 255, ${
      lineToQuestionCount[idx - 1] / maxQuestionCount
    })`;

    lineNumSpan.classList.add("line-number");
    uberSpan.append(lineNumSpan);

    // Create a span w/ the line of code
    let span = document.createElement("span");
    span.innerHTML = `${line}\n`;
    span.classList.add("actual-code-line");

    // Add the span to the codebox
    uberSpan.append(span);

    // Highlight on-hover...
    if (lineToQuestionCount[idx-1] > 0) {
      // uberSpan.classList.add("has-questions");
      span.classList.add("has-questions");
    }

    // Syntax highlighting. Note: this has to happen inside the span
    // (i.e., can't just do it on the codebox).
    hljs.highlightElement(span);

    // Example of how to add a click listener to a line of code.
    span.addEventListener("click", () => {
      console.log("HI");
      updateSidebar(idx);
    });
  });
}

function updateSidebar(lineOfCode) {
  let sidebar = document.getElementById("sidebar");
  sidebar.innerHTML = `<h2>Questions for line ${lineOfCode}</h2><hr class="thick-boi">`;

  DATA.Questions.forEach((el, idx) => {
    let [lo, hi] = el.Lines;
    if (lineOfCode < lo || lineOfCode > hi) return;  // Question doesn't include this line.

    let accord = document.createElement("div");
    accord.classList.add("accordion");
    accord.id = `accordionExample${idx}`;
    for (let i = 0; i < el.Questions.length; i++) {
      let thingy = document.createElement("div");
      thingy.classList.add("accordion-item");
      let ques = el.Questions[i];
      let short = ques;
      let maxLength = 50;
      if (ques.length > maxLength) {
        short = ques.slice(0, maxLength) + "...";
      }
      if (el.Lines[0] != el.Lines[1] && el.Lines[1] - el.Lines[0] < 40) {
        lines = `Lines ${el.Lines[0]} - ${el.Lines[1]}<hr class="thin-boi">`;
      } else if (el.Lines[0] == el.Lines[1]) {
        lines = '';
      }
      let ans = el.Chatgpt_response[i];
      thingy.innerHTML = `<h5><button class="accordion-button collapsed question-butt" type="button" data-bs-toggle="collapse" data-bs-target="#collapse${idx}_${i}" aria-expanded="false" aria-controls="collapseOne">
                    ${short} </button> </h5>
                    <div id="collapse${idx}_${i}" class="accordion-collapse collapse" aria-labelledby="headingOne">
                        <div class="accordion-body">
                        <div class="question-text p-2"><span class="line-chip">${lines}</span>${ques}</div> <div class="answer-text p-2"> ${ans} </div>
                        </div>
                    </div>`;
      accord.append(thingy);
    }
    let sep = document.createElement("div");
    sep.innerHTML = `<hr class="thick-boi">`;
    sidebar.append(accord);
    sidebar.append(sep);

    // Finally, let's highlight the code if we hover over the accordian :)
    accord.addEventListener("mouseover", () => {
        for (let i = lo; i <= hi; i++) {
            let line = document.getElementById(`L${i}`);
            line.classList.add("highlight");
        }
    });
    accord.addEventListener("mouseout", () => {
        for (let i = lo; i <= hi; i++) {
            let line = document.getElementById(`L${i}`);
            line.classList.remove("highlight");
        }
    });
  });
  const button = document.createElement("button");
  button.textContent = "Go Back";
  button.classList.add("back-button");
  sidebar.append(button);
  button.addEventListener("click", () => {
    loadSidebar(DATA);
  });
}

function loadSidebar(data) {
  let sidebar = document.getElementById("sidebar");
  sidebar.innerHTML = `<h2>Questions Summary<hr class="thick-boi">`;
  let ul = document.createElement("ul");
  DATA.Summary.forEach((el) => {
    let li = document.createElement("li");
    li.innerHTML = el;
    ul.appendChild(li);
  });
  sidebar.append(ul);
}
