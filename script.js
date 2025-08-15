(function(){
  const saved = localStorage.getItem('theme'); if(saved==='light') document.body.classList.add('light');
  document.getElementById('themeToggle')?.addEventListener('click', ()=>{
    document.body.classList.toggle('light');
    localStorage.setItem('theme', document.body.classList.contains('light')?'light':'dark');
  });
  if('serviceWorker' in navigator){
    navigator.serviceWorker.register('sw.js').catch(()=>{});
  }
})();