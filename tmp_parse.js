const fs = require("fs");

const raw = fs.readFileSync(process.argv[2], "utf8");

// Mirrors the frame loop in frontend/script.js exactly.
let buffer = raw;
const seen = [];
let stream = "";
let answer = "";
let phaseCount = 0;

let cut = buffer.indexOf("\n\n");

while (cut !== -1) {
    const frame = buffer.slice(0, cut);
    buffer = buffer.slice(cut + 2);

    let name = "message";
    const data = [];

    frame.split("\n").forEach((line) => {
        if (line.startsWith("event:")) {
            name = line.slice(6).trim();
        } else if (line.startsWith("data:")) {
            data.push(line.slice(5).trim());
        }
    });

    let payload = {};
    try {
        payload = JSON.parse(data.join("\n") || "{}");
    } catch (err) {
        payload = {};
    }

    seen.push(name);

    if (name === "subtask_phase") {
        phaseCount += 1;
    } else if (name === "subtask_delta") {
        stream += payload.text || "";
    } else if (name === "merged") {
        answer = payload.answer || "";
    } else if (name === "done") {
        answer = payload.answer || answer;
    }

    cut = buffer.indexOf("\n\n");
}

const counts = {};
seen.forEach((n) => {
    counts[n] = (counts[n] || 0) + 1;
});

console.log("events        :", JSON.stringify(counts));
console.log("phase updates :", phaseCount, "(UI needs > 0 to prove live steps)");
console.log("leftover bytes:", buffer.length, "(must be 0 or framing is broken)");
console.log("streamed text :", JSON.stringify(stream.slice(0, 60)));
console.log("final answer  :", JSON.stringify(answer.slice(0, 60)));
console.log("stream matches answer:", stream.trim() === answer.trim());
