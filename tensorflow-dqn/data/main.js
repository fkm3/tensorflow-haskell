'use strict';

document.addEventListener('DOMContentLoaded', () => {
  let img = document.getElementById('img');
  setInterval(() => { img.src = '/screen#' + new Date().getTime(); }, 50);
});
