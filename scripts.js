// static/js/scripts.js

function sendMessage() {
    var userText = document.getElementById("userInput").value;
    document.getElementById("response").innerHTML += "<div>User: " + userText + "</div>";
    fetch(`/get?msg=${userText}`)
        .then(response => response.text())
        .then(data => {
            document.getElementById("response").innerHTML += "<div>Bot: " + data + "</div>";
            document.getElementById("userInput").value = '';
        }); 
}
