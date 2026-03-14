async function sendMessage(){

let input = document.getElementById("userInput");
let message = input.value;

if(message.trim() === "") return;

let chatbox = document.getElementById("chatbox");

// user message
chatbox.innerHTML += `
<div class="item right">
<div class="msg"><p>${message}</p></div>
</div>
<br clear="both">
`;

input.value="";

// send request to FastAPI
let response = await fetch("/ask",{
method:"POST",
headers:{
"Content-Type":"application/json"
},
body:JSON.stringify({
question:message
})
});

let data = await response.json();

// bot message
chatbox.innerHTML += `
<div class="item">
<div class="icon"><i class="fa fa-user"></i></div>
<div class="msg"><p>${data.answer}</p></div>
</div>
<br clear="both">
`;

chatbox.scrollTop = chatbox.scrollHeight;
}