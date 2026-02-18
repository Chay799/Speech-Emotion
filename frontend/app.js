let mediaRecorder;
let audioChunks = [];
let timerInterval;
let seconds = 0;

function updateTimer(){
    seconds++;
    let min = String(Math.floor(seconds/60)).padStart(2,'0');
    let sec = String(seconds%60).padStart(2,'0');
    document.getElementById("recordTimer").innerText = `${min}:${sec}`;
}

// ================= UPLOAD FILE =================
async function predictUpload(){
    let file = document.getElementById("fileInput").files[0];
    sendAudioToAPI(file);
}

// ================= RECORD AUDIO =================
async function startRecording(){

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);

    // ⭐ SHOW RECORDER UI
    seconds = 0;
    document.getElementById("recordCircle").style.display = "block";
    document.getElementById("recordStatus").innerText = "Recording...";
    document.getElementById("recordTimer").innerText = "00:00";
    timerInterval = setInterval(updateTimer,1000);

    mediaRecorder.start();
    audioChunks = [];

    mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
}

async function stopRecording(){

    mediaRecorder.stop();

    // ⭐ STOP UI
    clearInterval(timerInterval);
    document.getElementById("recordCircle").style.display = "none";
    document.getElementById("recordStatus").innerText = "Processing...";

    mediaRecorder.onstop = async () => {
        let blob = new Blob(audioChunks, { type: "audio/wav" });
        sendAudioToAPI(blob);
        document.getElementById("recordStatus").innerText = "Idle";
    };
}

// ================= SEND TO FASTAPI =================
async function sendAudioToAPI(file){

    let formData = new FormData();
    formData.append("file", file);

    let res = await fetch("/predict", {
        method:"POST",
        body:formData
    });

    let data = await res.json();
    console.log("API RESPONSE =", data);
    
    // ⭐ SHOW DETECTED LANGUAGE
    document.getElementById("languageText").innerHTML = "🌐 Language: " + data.language;


    // TEXT RESULT
    document.getElementById("bigEmotion").innerHTML = "🎯 " + data.emotion;
    document.getElementById("confidenceText").innerHTML =
        "Confidence: " + (data.confidence*100).toFixed(1) + "%";

    // BARS
    const neutralBar = document.getElementById("neutralBar");
    const angryBar = document.getElementById("angryBar");
    const sadBar   = document.getElementById("sadBar");

    neutralBar.style.width = (data.probabilities.neutral * 100) + "%";
    angryBar.style.width = (data.probabilities.angry * 100) + "%";
    sadBar.style.width   = (data.probabilities.sad   * 100) + "%";
}

// ⭐ SHOW AUDIO LENGTH WHEN UPLOADED
document.getElementById("fileInput").addEventListener("change", function(){
    const file = this.files[0];
    if(!file) return;

    const audio = new Audio(URL.createObjectURL(file));
    audio.addEventListener("loadedmetadata", () => {
        document.getElementById("confidenceText").innerHTML =
            "Audio length: " + audio.duration.toFixed(1) + " sec";
    });
});
