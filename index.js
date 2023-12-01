var testCode;
var data;

window.onload = () => {
  // testCode is in data.js.
  hackyPreprocessing();
  loadCode(testCode);
};

function loadCode(codeText) {
  let codebox = document.getElementById("da-code");
  codebox.innerHTML = "";
  loadSummary(data);

  let lineToQuestionCount = testCode.split("\n").map((_, idx) => {
    let lineNum = idx + 1;
    return data.Questions.filter((q) => {
      let [lo, hi] = q.Lines;
      return lo <= lineNum && lineNum <= hi;
    }).length;
  });
  // lineToQuestionCount.forEach((count, idx) => {
  //   console.log(`Line ${idx + 1} has ${count} questions.`);
  // });
  let maxQuestionCount = Math.max(...lineToQuestionCount);

  // For each line of code
  testCode.split("\n").forEach((line, idx) => {
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

  data.Questions.forEach((el, idx) => {
    let [lo, hi] = el.Lines;
    if (lineOfCode < lo || lineOfCode > hi) return;  // Question doesn't include this line.

    let accord = document.createElement("div");
    accord.classList.add("accordion");
    accord.id = `accordionExample${idx}`;
    for (let i = 0; i < el.Questions.length; i++) {
      let ques = el.Questions[i];
      let ans = el.Chatgpt_response[i];
      ans = wrapWithPTags(ans);

      let thingy = document.createElement("div");
      thingy.classList.add("accordion-item");
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
      // let ans = el.Chatgpt_response[i];
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
        document.getElementById(`L${lo}`).scrollIntoView({behavior: "smooth", block: "center"});
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
    loadSidebar(data);
  });
}

// Initial sidebar (i.e., the summaries)
function loadSummary(data) {
  let sidebar = document.getElementById("sidebar");
  sidebar.innerHTML = `<h2>Questions Summary<hr class="thick-boi">`;
  let ul = document.createElement("ul");
  data.Summary.forEach(({daText}, idx) => {
    let li = document.createElement("li");
    let a = document.createElement("a");
    a.href = "#";
    a.classList.add("link-info");
    li.append(a);
    a.innerHTML = daText;
    ul.appendChild(li);
    a.addEventListener("click", (e) => {
      e.preventDefault();
      loadSummaryDetails(idx);
    });
  });
  sidebar.append(ul);
}

// idx: the idx of which summary doob we want to get the q's for lol.
function loadSummaryDetails(idx) {
  let sidebar = document.getElementById("sidebar");
  sidebar.innerHTML = `<h2>Summary Summary<hr class="thick-boi">`;

  let p = document.createElement("p");
  p.innerHTML = data.Summary[idx].daText;
  sidebar.append(p);

  // This is basically copy-pasta from updateSidebar lol.
  data.Summary[idx].questionIdxs.forEach((idx) => {
    let [lo, hi] = data.Questions[idx].Lines;
    let linesText = lo === hi ? `(Line ${lo})` : `(Lines ${lo}-${hi})`;

    let accord = document.createElement("div");
    accord.classList.add("accordion");
    accord.id = `accordionExample${idx}`;

    let el = data.Questions[idx];
    for (let i = 0; i < el.Questions.length; i++) {
      let ques = el.Questions[i];
      let quesLines = ques + " " + linesText;
      let ans = el.Chatgpt_response[i];
      ans = wrapWithPTags(ans);

      let thingy = document.createElement("div");
      thingy.classList.add("accordion-item");
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
      // let ans = el.Chatgpt_response[i];
      thingy.innerHTML = `<h5><button class="accordion-button collapsed question-butt" type="button" data-bs-toggle="collapse" data-bs-target="#collapse${idx}_${i}" aria-expanded="false" aria-controls="collapseOne">
                    ${short} </button> </h5>
                    <div id="collapse${idx}_${i}" class="accordion-collapse collapse" aria-labelledby="headingOne">
                        <div class="accordion-body">
                        <div class="question-text p-2"><span class="line-chip">${lines}</span>${quesLines}</div> <div class="answer-text p-2"> ${ans} </div>
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
        document.getElementById(`L${lo}`).scrollIntoView({behavior: "smooth", block: "center"});
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
    loadSummary(data);
  });
}

function wrapWithPTags(str) {
  return str.split("\n\n").map((line) => `<p>${line}</p>`).join("\n");
}


function hackyPreprocessing() {
  // Trim the first two lines of testCode lol.
  testCode = TEST_CODE.substring(2);

  // Subtract 2 from each line number in DATA.
  DATA.Questions.forEach((el) => {
    el.Lines = el.Lines.map((x) => x - 2);
  });
  data = DATA;
}