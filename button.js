
const selector = ".table-container img, .table-container p";


window.addEventListener("DOMContentLoaded", () => {
  const items = document.querySelectorAll(selector);
  items.forEach(el => {
    
    el.classList.add("fade", "hidden");
    
    el.style.display = "none";
  });
});


function run() {
  const items = document.querySelectorAll(selector);

  items.forEach(el => {
    if (el.classList.contains("hidden")) {
    
      el.style.display = "";         
      setTimeout(() => el.classList.remove("hidden"), 10);
    } else {
      
      el.classList.add("hidden");
    
      setTimeout(() => { el.style.display = "none"; }, 300); 
    }
  });
}
