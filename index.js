window.onload = () => {
    // TEST_CODE is in data.js.
    loadCode(TEST_CODE);
};


function loadCode(codeText) {
    let codebox = document.getElementById("da-code");
    codebox.innerHTML = "";

    // For each line of code
    TEST_CODE.split("\n").forEach((line, idx) => {
        idx++;

        // Create a span w/ the line of code
        let span = document.createElement("span");
        span.id = `L${idx}`;
        span.innerHTML = `${line}\n`;
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
}

function updateSidebar(lineOfCode) {
    let sidebar = document.getElementById("sidebar");
    sidebar.innerHTML = `<h2>You clicked line: ${lineOfCode}!</h2><hr><hr>`;
    
    DATA.Questions.forEach(el => {
        if(lineOfCode >= el.Lines[0] && lineOfCode <= el.Lines[1])
        {
            let accord = document.createElement("div");
            accord.classList.add("accordion");
            for(let i = 0; i <el.Questions.length; i++)
            {
                let thingy = document.createElement("div");
                thingy.classList.add("accordion-item");
                let ques = el.Questions[i];
                let short = ques;
                if(ques.length > 25)
                {   
                    short = ques.slice(0,25) + "...";
                }
                let ans = el.Chatgpt_response[i]
                let div = document.createElement("div")
                div.innerHTML = `<h3><button class="accordion-button" type="button" data-bs-toggle="collapse" data-bs-target="#collapseOne" aria-expanded="false" aria-controls="collapseOne">
                    ${short} </button> </h3>
                    <div id="collapseOne" class="accordion-collapse collapse show" aria-labelledby="headingOne" data-bs-parent="#accordionExample">
                        <div class="accordion-body">
                        <strong>${ques}</strong><br> ${ans} <br>
                        </div>
                    </div>`;
                thingy.append(div);
                accord.append(thingy);
            }
            let sep = document.createElement("div");
            sep.innerHTML = `<hr><hr>`;
            sidebar.append(accord);
            sidebar.append(sep);
        }
    });
    
}
