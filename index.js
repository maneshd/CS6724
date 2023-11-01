window.onload = () => {
    // TEST_CODE is in data.js.
    loadCode(TEST_CODE);
};


function loadCode(codeText) {
    let codebox = document.getElementById("da-code");
    codebox.innerHTML = "";
    loadSidebar(DATA);

    // For each line of code
    TEST_CODE.split("\n").forEach((line, idx) => {
        idx++;

        // Create a span w/ the line of code
        let span = document.createElement("span");
        span.id = `L${idx}`;
        let linenum = String("   " + idx).slice(-3)
        span.innerHTML = `${linenum}  ${line}\n`;
        span.classList.add("loc-span");

        // Add the span to the codebox
        codebox.append(span);

        // Highlight the code. Note: this has to happen inside the span
        // (i.e., can't just do it on the codebox).
        hljs.highlightElement(span);

        // Example of how to add a click listener to a line of code.
        span.addEventListener("click", () => {
            console.log("HI");
            updateSidebar(idx);
        });
    });
    readJSON();
}

function loadSidebar(data){
    let sidebar = document.getElementById("sidebar");
    sidebar.innerHTML = `<h2>Questions Summary<hr>`;
    let ul = document.createElement("ul")
    DATA.Summary.forEach(el => {
        let li = document.createElement('li')
        li.innerHTML = el
        ul.appendChild(li)
    });
    sidebar.append(ul)
}

function updateSidebar(lineOfCode) {
    let sidebar = document.getElementById("sidebar");
    let button = document.createElement("button");
    //button.innerHTML = 'class=\'btn btn-success pull-right\'> Button Text';
    codebox.append(button);
    sidebar.innerHTML = `<h2>You clicked line: ${lineOfCode}!</h2><hr>`;

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