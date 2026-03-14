async function sendMessage(){

let input = document.getElementById("userInput");
let message = input.value;

if(message.trim() === "") return;

let chatbox = document.getElementById("chatbox");

// show user message
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

// create bot message container
let botItem = document.createElement("div");
botItem.className = "item";

botItem.innerHTML = `
<div class="icon"><i class="fa fa-user"></i></div>
<div class="msg"><p></p></div>
`;

chatbox.appendChild(botItem);

let botText = botItem.querySelector("p");

// read streaming response
const reader = response.body.getReader();
const decoder = new TextDecoder();

let botMessage = "";

while(true){

const {done, value} = await reader.read();

if(done) break;

botMessage += decoder.decode(value);

botText.textContent = botMessage;

chatbox.scrollTop = chatbox.scrollHeight;
}
}