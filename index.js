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
    sidebar.innerHTML = `You clicked line: ${lineOfCode}!`;
}