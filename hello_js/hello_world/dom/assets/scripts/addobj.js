const bodyElement = document.querySelector("body");

const newbe = document.createElement("p");
newbe.textContent = "I exist or not"

// bodyElement.appendChild(newbe);

//like insertAfter
bodyElement.insertBefore(newbe, bodyElement.children[0].nextSibling);

const clonedNode = newbe.cloneNode(true);
bodyElement.insertBefore(clonedNode, newbe);

// setTimeout(function () {
//   bodyElement.insertBefore(newbe, bodyElement.children[0]);
// }, 10000);
