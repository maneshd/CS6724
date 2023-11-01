window.onload = () => {
    // TEST_CODE is in data.js.
    loadCode(TEST_CODE);
};


function loadCode(codeText) {
    let codebox = document.getElementById("da-code");
    codebox.innerHTML = "";

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
        lineNumSpan.style.backgroundColor = `rgba(205, 0, 255, ${lineToQuestionCount[idx - 1] / (maxQuestionCount*2)})`;
        lineNumSpan.classList.add("line-number")
        uberSpan.append(lineNumSpan);

        // Create a span w/ the line of code
        let span = document.createElement("span");
        span.innerHTML = `${line}\n`;
        span.classList.add("actual-code-line");


        // Add the span to the codebox
        uberSpan.append(span);

        // Syntax highlighting. Note: this has to happen inside the span
        // (i.e., can't just do it on the codebox).
        hljs.highlightElement(span);

        // Example of how to add a click listener to a line of code.
        span.addEventListener("click", () => {
            console.log("HI");
            updateSidebar(idx);
        });

    });
    readJSON()
}

function updateSidebar(lineOfCode) {
    let sidebar = document.getElementById("sidebar");
    sidebar.innerHTML = `<h2>You clicked line: ${lineOfCode}!</h2><hr><hr>`;

    DATA.Questions.forEach(el => {
        if(lineOfCode >= el.Lines[0] && lineOfCode <= el.Lines[1])
        {
            for(let i = 0; i < el.Questions.length; i++)
            {
                let ques = el.Questions[i]
                let ans = el.Chatgpt_response[i]
                let lines = ``
                //40 is an arbitrary number, so we do not have a bunch of questions with "1-72" when someone highlights all the code
                if (el.Lines[0] != el.Lines[1] && el.Lines[1] - el.Lines[0] < 40){
                    lines = `Lines ${el.Lines[0]} - ${el.Lines[1]} <br>`
                } 
                let div = document.createElement("div")
                div.innerHTML = `${lines} ${ques} <br><hr> ${ans} <hr><hr>`;
                div.classList.add("QA")
    
                sidebar.append(div)
            }
        }
    });
    
}